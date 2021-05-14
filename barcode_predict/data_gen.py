import numpy as np
from PIL import Image, ImageDraw
import pickle
import os
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp
import time
import cv2


class DataGen:

    def __init__(
            self,
            datadir="/home/dmytro/Data/scanner_images",
            datadir_val="/home/dkorenkevych/Data/scanner_images_test",
            num_workers=10,
            img_dim=(256, 256),
            separate_validation=False
    ):
        self.datadir = datadir
        self.datadir_val = datadir_val
        self.separate_validation = separate_validation
        self._img_keys = {
            'train': [],
            "val": []
        }
        self._outlines = {
            'train': [],
            "val": []
        }
        self._labels = {
            'train': {},
            'val': {}
        }
        self.batch_size = {
            'train': 32,
            'val': 32
        }
        self.data_size = {
            'train': 0,
            'val': 0
        }
        self.current_batch_id = {
            'train': 0,
            'val': 0
        }
        self.img_dim = img_dim
        self._batch_buffer_size = 10
        self._terminate = mp.Value('i', 0)
        self._start = {
            'train': mp.Value('i', 0),
            'val': mp.Value('i', 0)
        }
        self.progress_in_epoch = mp.Value('d', 0.0)
        self._build_dataset()
        self._init_shared_variables()
        self.separate_validation = separate_validation
        #self._compute_mean_statistics()
        self._start_workers(num_workers)

    def _init_shared_variables(self):
        """Initializes shared arrays and shared variables."""
        channels = 1
        self.full_dim = self.img_dim + (channels,)
        self._new_batch = {'train': {}, 'val': {}}
        self._batch_dict = {'train': {}, 'val': {}}
        self._locks = {'train': {}, 'val': {}}
        self._ix_locks = {'train': mp.Lock(), 'val': mp.Lock()}
        self._permuted_ix = {
            'train': np.frombuffer(mp.Array('i', self.data_size['train']).get_obj(), dtype="int32"),
            'val': np.frombuffer(mp.Array('i', self.data_size['val']).get_obj(), dtype="int32")
        }
        self._ix_processed = {
            'train': np.frombuffer(mp.Array('i', self.data_size['train']).get_obj(), dtype="int32"),
            'val': np.frombuffer(mp.Array('i', self.data_size['val']).get_obj(), dtype="int32")
        }
        self.manager = mp.Manager()
        self.processed_batches = {}
        self.unprocessed_batches = {}
        self.unprocessed_batches_local = {}
        for dataset in ["train", "val"]:
            ix = np.random.permutation(self.data_size[dataset]).astype('int32')
            np.copyto(self._permuted_ix[dataset], ix)
            np.copyto(self._ix_processed[dataset], np.ones_like(self._ix_processed[dataset]))
            batch_arr_size = self.batch_size[dataset] * np.product(self.full_dim)
            batch_shape = (self.batch_size[dataset],) + self.full_dim
            self.processed_batches[dataset] = self.manager.dict()
            self.unprocessed_batches[dataset] = self.manager.dict()
            self.unprocessed_batches_local[dataset] = self.manager.dict()
            for i in range(0, self.data_size[dataset], self.batch_size[dataset]):
                self.unprocessed_batches[dataset][i] = 1
                self.unprocessed_batches_local[dataset][i] = 1
            for i in range(self._batch_buffer_size):
                self._new_batch[dataset][i] = mp.Value('i', 0)
                data_arr = np.frombuffer(mp.Array('f', int(batch_arr_size)).get_obj(), dtype="float32")
                data_arr = data_arr.reshape(batch_shape)
                labels_arr = np.frombuffer(mp.Array('f', int(batch_arr_size)).get_obj(), dtype="float32")
                labels_arr = labels_arr.reshape(batch_shape)
                self._batch_dict[dataset][i] = {'data': data_arr, "labels": labels_arr, "start_ix": mp.Value('i', 0)}
                self._locks[dataset][i] = mp.Lock()

    def _start_workers(self, num_workers):
        """Starts concurrent processes that build data minibatches."""
        self._process_list = []
        for i in range(num_workers):
            p = mp.Process(target=self.prepare_minibatch, args=('train', i))
            p.start()
            p_val = mp.Process(target=self.prepare_minibatch, args=('val', i))
            p_val.start()
            self._process_list.append(p)
            self._process_list.append(p_val)

    def _build_dataset(self, val_split=0.1):
        with open(os.path.join(self.datadir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        # if len(metadata['image_file']) > 1000000:
        #     val_split = 0.01
        self.val_metadata = metadata
        if self.separate_validation:
            with open(os.path.join(self.datadir_val, "metadata.pkl"), "rb") as f:
                self.val_metadata = pickle.load(f)
        self.data_size['val'] = int(np.clip(int(len(metadata['image_file']) * val_split), 0, 10000))
        self.data_size['train'] = len(metadata['image_file']) - self.data_size['val']
        self.data_size['train'] = self.data_size['train'] - self.data_size['train'] % self.batch_size['train']
        self._img_keys['train'] = metadata['image_file'][:self.data_size['train']]
        self._outlines['train'] = metadata['barcode_outline'][:self.data_size['train']]
        self.data_size['val'] = self.data_size['val'] - self.data_size['val'] % self.batch_size['val']
        self._img_keys['val'] = metadata['image_file'][-self.data_size['val']:]
        self._outlines['val'] = metadata['barcode_outline'][-self.data_size['val']:]
        if self.separate_validation:
            self.data_size['val'] = len(self.val_metadata['image_file'])
            self.data_size['val'] = self.data_size['val'] - self.data_size['val'] % self.batch_size['val']
            self._img_keys['val'] = self.val_metadata['image_file'][-self.data_size['val']:]
            self._outlines['val'] = self.val_metadata['barcode_outline'][-self.data_size['val']:]
            for key in self.val_metadata:
                self.val_metadata[key] = self.val_metadata[key][-self.data_size['val']:]
        print("data size", self.data_size)

    def _compute_mean_statistics(self):
        if os.path.isfile("pixel_mean.pkl"):
            with open("pixel_mean.pkl", "rb") as f:
                self.pixel_mean = pickle.load(f)
            return
        print("computing mean")
        self.pixel_mean = np.zeros(self.full_dim)
        count = 0
        for i in range(len(self._img_keys['train'])):
            filename = self._img_keys['train'][i]
            img = self.load_image(filename, outline, 'train')
            self.pixel_mean += img
            count += 1
        self.pixel_mean /= count
        with open('pixel_mean.pkl', "wb") as f:
            pickle.dump(self.pixel_mean, f)
        print("done")

    def buffer_empty(self, dataset):
        for id in self._new_batch[dataset]:
            if self._new_batch[dataset][id].value:
                return False
        return True

    def prepare_minibatch(self, dataset="train", proc_id=0):
        """Builds minibatches and stores them to shared memory.

        This function is run by concurrent processes.
        """

        if dataset == "val":
            cvb = 1
        scan_attempt_start = time.time()
        while not self._terminate.value:
            for id in self._new_batch[dataset]:
                if not self._new_batch[dataset][id].value:
                    locked = self._locks[dataset][id].acquire(False)
                    if not locked:
                        continue
                    locked2 = False
                    start_time = time.time()
                    while not locked2:
                        locked2 = self._ix_locks[dataset].acquire()
                        if not locked2:
                            time.sleep(0.01)
                    if not locked2:
                        print("failed to get a lock")
                        continue
                    # if time.time() - start_time > 1e-3:
                    #     print("proc {} waited for lock2 {}".format(proc_id, time.time() - start_time))
                    #if self.data_size[dataset] - self._start[dataset].value < self.batch_size[dataset]:
                    if not len(self.unprocessed_batches[dataset]):
                        print("Reached the end of the dataset {} resetting".format(dataset))
                        #np.copyto(self._ix_processed[dataset], np.zeros(self._ix_processed[dataset].shape, dtype="int32"))
                        self._start[dataset].value = 0
                        wait_start = time.time()
                        while len(self.unprocessed_batches_local[dataset]) and time.time() - wait_start < 60:
                            time.sleep(0.01)
                        if time.time() - wait_start > 60:
                            print("Resetting by timeout")
                        ix = np.random.permutation(self.data_size[dataset]).astype('int32')
                        np.copyto(self._permuted_ix[dataset], ix)
                        for i in range(0, self.data_size[dataset], self.batch_size[dataset]):
                            self.unprocessed_batches[dataset][i] = 1
                            self.unprocessed_batches_local[dataset][i] = 1
                        print("Finished resetting {}".format(dataset))
                    #np.copyto(self._ix_processed[dataset][self._start[dataset]: self._start[dataset] + self.batch_size[dataset]], np.ones(self.batch_size[dataset], dtype="int32"))
                    for key in self.unprocessed_batches[dataset].keys():
                        start_ix = key
                        del self.unprocessed_batches[dataset][key]
                        break
                    ix_range = self._permuted_ix[dataset][start_ix: start_ix + self.batch_size[dataset]]
                    self._ix_locks[dataset].release()
                    if len(ix_range) < self.batch_size[dataset]:
                        continue
                    #load_start = time.time()
                    data, labels = self.load_batch(ix_range, dataset=dataset)
                    #print("proc id {} loaded data in {}".format(proc_id, time.time() - load_start))
                    data = np.array(data)
                    #data = np.expand_dims(np.array(data), axis=-1)
                    labels = np.array(labels)
                    np.copyto(self._batch_dict[dataset][id]["data"], data)
                    np.copyto(self._batch_dict[dataset][id]["labels"], labels)
                    self._batch_dict[dataset][id]["start_ix"].value = start_ix
                    self._new_batch[dataset][id].value = 1
                    #print("stored batch into id", id)
                    batches_aval = sum([self._new_batch[dataset][i].value for i in self._new_batch[dataset]])
                    #print("start", dataset, self._start[dataset].value)
                    if dataset == "train" and batches_aval < 5:
                        print("Prepared new batch, {} batches available".format(batches_aval))
                    self.progress_in_epoch.value = (self._start[dataset].value + 0.0)/len(self._permuted_ix[dataset])
                    #break
                    self._locks[dataset][id].release()
                    #print("proc_id {} finished scan attempt in {}".format(proc_id, time.time() - scan_attempt_start))
                    scan_attempt_start = time.time()
            #print("Im sleeping")
            time.sleep(0.1)

    def generate_batch(self, dataset="train"):
        """A generator reading and returning data batches from shared memory.

        To be used by an external training function.
        """
        while True:
            for id in self._new_batch[dataset]:
                if self._new_batch[dataset][id].value:
                    #print("reading batch from id", id)
                    #ix = np.random.permutation(self._batch_dict[dataset][id]["data"].shape[0])
                    data = np.copy(self._batch_dict[dataset][id]["data"])
                    labels = np.copy(self._batch_dict[dataset][id]["labels"])
                    self._new_batch[dataset][id].value = 0
                    if not self._batch_dict[dataset][id]["start_ix"].value in self.unprocessed_batches_local[dataset]:
                        print("ERRROR, {} not in unprocessed".format(self._batch_dict[dataset][id]["start_ix"].value))
                    else:
                        del self.unprocessed_batches_local[dataset][self._batch_dict[dataset][id]["start_ix"].value]
                    #print("Left unprocessed local", len(self.unprocessed_batches_local[dataset]))
                    yield data, labels

    def load_batch(self, ix, dataset='train'):
        """Loads a batch of images defined by indices in ix list."""
        images = []
        labels = []
        for i in ix:
            filename, outline = self._img_keys[dataset][i], self._outlines[dataset][i]
            img, label = self.load_image(filename, outline, dataset)
            images.append(np.expand_dims(img, axis=-1))
            #img[0, 0, 0] = i + 0.0
            labels.append(np.expand_dims(label, axis=-1).astype("float32"))
            # if dataset == "val":
            #     plt.figure(figsize=(10, 2.8))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(img, cmap='gray')
            #     plt.xticks([])
            #     plt.yticks([])
            #
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(label.astype("float32"), cmap='gray')
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.subplots_adjust(left=0, right=1)
            #     plt.show()
        return images, labels

    def preprocess_image(self, img, dataset):
        # if not img.shape == self.img_dim:
        #     img = cv2.resize(img, self.img_dim, cv2.INTER_AREA)
        img = (img/255.0).astype('float32')
        #img -= np.mean(img)
        if True or not dataset == "train":
            return img
        shift = np.random.randint(0, self.crop_size, size=(2))
        new_img = np.zeros(tuple(np.array(img.shape) + self.crop_size), dtype="float32")
        new_img[self.crop_size//2:-self.crop_size//2, self.crop_size//2:-self.crop_size//2] = img
        return new_img[shift[0]:shift[0] + img.shape[0], shift[1]:shift[1] + img.shape[1]]

    def load_image(self, filename, outline, dataset):
        # if dataset == "val" and self.separate_validation:
        #     filename = filename.replace("scanner_images", "scanner_images_test")
        img = np.load(filename)
        img = self.preprocess_image(img, dataset)
        # with open(filename, "rb") as f:
        #     jpg_frames = pickle.load(f)
        # img, label_orig = [np.array(Image.open(jpg)) for jpg in jpg_frames]
        pil_mask = Image.new('L', img.shape, 0)
        outline_arr = [(point["x"], point["y"]) for point in outline]
        ImageDraw.Draw(pil_mask).polygon(outline_arr, outline=1, fill=1)
        label = np.array(pil_mask)
        #print("labels close", np.allclose(label, label_orig))
        #label[label > 0] = 1
        return img, label


if __name__ == "__main__":
    d = DataGen()
    for mb in d.generate_batch("train"):
        data, labels = mb
