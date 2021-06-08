import numpy as np
from PIL import Image, ImageDraw
import pickle
import os
import multiprocessing as mp
import time
import cv2


class BatchGen:
    """Generates minibatches of data to be used by a training function.

    Pre-fetches minibatches of images saved to hard disk using
    multiprocessing to minimize the bottleneck of loading the data.

    Args:
        datadir: a path to a directory where the data is stored.
        datadir_val: an optinal path to a validation data directory,
            only used if the separate_validation flag is True.
        num_workers: numer of processes to spawn to build data batches
            in parallel.
        img_dim: a tuple target size of the images (the images will be
            resized if their current size is different).        
        separate_validation: whether to use a separately stored dataset
            for validation, otherwise the main dataset will be split into
            training and validation parts.
        use_random_crops: whether to apply random crop data augmentation
            during training.
        crop_size: an integer representing max margin in pixels for
            a random crop (used only if use_random_crops is True).
    """

    def __init__(
            self,
            datadir="/home/dmytro/Data/scanner_images",
            datadir_val="/home/dkorenkevych/Data/scanner_images_test",
            num_workers=10,
            img_dim=(256, 256),
            crop_size=50,
            separate_validation=False,
            use_random_crops=False
    ):
        self.datadir = datadir
        self.datadir_val = datadir_val
        self.separate_validation = separate_validation
        self._use_random_crops = use_random_crops
        self.img_dim = img_dim
        self._batch_buffer_size = 10
        self._build_dataset()
        self._init_vars()
        self._init_shared_vars()               
        self._start_workers(num_workers)

    def _init_shared_vars(self):
        """Initializes necessary shared arrays and shared variables.

        Initializes variables necessary to communicate between processes.
        All the data is exchanged using shared memory for the best performance.
        """
        self._terminate = mp.Value('i', 0)
        self._locks = {'train': {}, 'val': {}}
        self._start = {
            'train': mp.Value('i', 0),
            'val': mp.Value('i', 0)
        }
        self.progress_in_epoch = mp.Value('d', 0.0)
        channels = 1
        self.full_dim = self.img_dim + (channels,)
        self._ix_locks = {'train': mp.Lock(), 'val': mp.Lock()}
        self._permuted_ix = {
            'train': np.frombuffer(
                mp.Array('i', self.data_size['train']).get_obj(),
                dtype="int32"
            ),
            'val': np.frombuffer(
                mp.Array('i', self.data_size['val']).get_obj(),
                dtype="int32"
            )
        }
        self._ix_processed = {
            'train': np.frombuffer(
                mp.Array('i', self.data_size['train']).get_obj(),
                dtype="int32"
            ),
            'val': np.frombuffer(
                mp.Array('i', self.data_size['val']).get_obj(),
                dtype="int32"
            )
        }
        self._manager = mp.Manager()
        self._processed_batches = {}
        self._unprocessed_batches = {}
        self._unprocessed_batches_local = {}
        for dataset in ["train", "val"]:
            ix = np.random.permutation(self.data_size[dataset]).astype('int32')
            np.copyto(self._permuted_ix[dataset], ix)
            np.copyto(
                self._ix_processed[dataset],
                np.ones_like(self._ix_processed[dataset])
            )
            batch_arr_size = self.batch_size[dataset] * np.product(self.full_dim)
            batch_shape = (self.batch_size[dataset],) + self.full_dim
            self._processed_batches[dataset] = self._manager.dict()
            self._unprocessed_batches[dataset] = self._manager.dict()
            self._unprocessed_batches_local[dataset] = self._manager.dict()
            for i in range(0, self.data_size[dataset], self.batch_size[dataset]):
                self._unprocessed_batches[dataset][i] = 1
                self._unprocessed_batches_local[dataset][i] = 1
            for i in range(self._batch_buffer_size):
                self._new_batch[dataset][i] = mp.Value('i', 0)
                data_arr = np.frombuffer(
                    mp.Array('f', int(batch_arr_size)).get_obj(),
                    dtype="float32"
                )
                data_arr = data_arr.reshape(batch_shape)
                labels_arr = np.frombuffer(
                    mp.Array('f', int(batch_arr_size)).get_obj(),
                    dtype="float32"
                )
                labels_arr = labels_arr.reshape(batch_shape)
                self._batch_dict[dataset][i] = {
                    'data': data_arr,
                    "labels": labels_arr,
                    "start_ix": mp.Value('i', 0)
                }
                self._locks[dataset][i] = mp.Lock()

    def _init_vars(self):
        """Initializes dictionaries and variables to manipulate the data."""
        self._img_keys = {'train': [], "val": []}
        self._outlines = {'train': [], "val": []}
        self._labels = {'train': {}, 'val': {}}
        self._new_batch = {'train': {}, 'val': {}}
        self._batch_dict = {'train': {}, 'val': {}}
        self.batch_size = {
            'train': 100,
            'val': 100
        }
        self.data_size = {
            'train': 0,
            'val': 0
        }
        self.current_batch_id = {
            'train': 0,
            'val': 0
        }

    def _start_workers(self, num_workers):
        """Starts concurrent processes that build data minibatches."""
        self._process_list = []
        for i in range(num_workers):
            p = mp.Process(target=self._prepare_minibatch, args=('train', i))
            p.start()
            p_val = mp.Process(target=self._prepare_minibatch, args=('val', i))
            p_val.start()
            self._process_list.append(p)
            self._process_list.append(p_val)

    def _build_dataset(self, val_split=0.1):
        """Builds datasets to use for minibatches generation.

        Args:
            val_split: a float representing data training/validation split
                ratio (only used if self.separate_validation flag is False)
        """
        with open(os.path.join(self.datadir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        self.val_metadata = metadata
        if self.separate_validation:
            with open(os.path.join(self.datadir_val, "metadata.pkl"), "rb") as f:
                self.val_metadata = pickle.load(f)
        val_size = np.clip(len(metadata['image_file']) * val_split, 0, 10000)
        self.data_size['val'] = int(val_size)
        train_size = len(metadata['image_file']) - self.data_size['val']
        train_size -= train_size % self.batch_size['train']        
        self.data_size['train'] = train_size        
        self._img_keys['train'] = metadata['image_file'][:self.data_size['train']]
        self._outlines['train'] = metadata['barcode_outline'][:self.data_size['train']]
        self.data_size['val'] -= self.data_size['val'] % self.batch_size['val']
        self._img_keys['val'] = metadata['image_file'][-self.data_size['val']:]
        self._outlines['val'] = metadata['barcode_outline'][-self.data_size['val']:]
        if self.separate_validation:
            self.data_size['val'] = len(self.val_metadata['image_file'])
            self.data_size['val'] -= self.data_size['val'] % self.batch_size['val']
            self._img_keys['val'] = self.val_metadata['image_file'][-self.data_size['val']:]
            self._outlines['val'] = self.val_metadata['barcode_outline'][-self.data_size['val']:]
            for key in self.val_metadata:
                self.val_metadata[key] = self.val_metadata[key][-self.data_size['val']:]
        print("data size", self.data_size)

    def _compute_mean_statistics(self):
        """Computes pixel-wise mean and st.d. of a training dataset."""
        if os.path.isfile("pixel_mean.pkl"):
            with open("pixel_mean.pkl", "rb") as f:
                self.pixel_mean = pickle.load(f)
            return
        print("computing mean")
        self.pixel_mean = np.zeros(self.full_dim)
        count = 0
        for i in range(len(self._img_keys['train'])):
            filename = self._img_keys['train'][i]
            outline = self._outlines['train'][i]
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
    
    def _acquire_locks(self, dataset, id):
        locked = self._locks[dataset][id].acquire(False)
        if not locked:
            return False
        locked_ix = False
        while not locked_ix:
            locked_ix = self._ix_locks[dataset].acquire()
            if not locked_ix:
                time.sleep(0.01)
        if not locked_ix:            
            return False
        return True
    
    def progress_in_epoch(self, dataset):
        return (self._start[dataset].value + 0.0) / len(self._permuted_ix[dataset])

    def _reset_epoch(self, dataset):
        print("Reached the end of the dataset {}, resetting".format(dataset))
        self._start[dataset].value = 0
        wait_start = time.time()
        while (len(self._unprocessed_batches_local[dataset])
               and time.time() - wait_start < 60):
            time.sleep(0.01)
        if time.time() - wait_start > 60:
            print("Resetting by timeout.")
        ix = np.random.permutation(self.data_size[dataset]).astype('int32')
        np.copyto(self._permuted_ix[dataset], ix)
        for i in range(0, self.data_size[dataset], self.batch_size[dataset]):
            self._unprocessed_batches[dataset][i] = 1
            self._unprocessed_batches_local[dataset][i] = 1
        print("Finished resetting {}".format(dataset))
        
    def _prepare_minibatch(self, dataset="train"):
        """Builds data minibatches and stores them to shared memory.

        This is the main function run by concurrent processes.
        
        Args:
            dataset: a dataset to build minibatches from ('train' or 'val').
        """
        while not self._terminate.value:
            for id in self._new_batch[dataset]:
                if not self._new_batch[dataset][id].value:
                    if not self._acquire_locks(dataset, id):
                        print("failed to get a lock")
                        continue
                    if not len(self._unprocessed_batches[dataset]):
                        self._reset_epoch(dataset)
                    for key in self._unprocessed_batches[dataset].keys():
                        start_ix = key
                        del self._unprocessed_batches[dataset][key]
                        break
                    ix_range = self._permuted_ix[dataset][start_ix: start_ix + self.batch_size[dataset]]
                    self._ix_locks[dataset].release()
                    if len(ix_range) < self.batch_size[dataset]:
                        continue
                    data, labels = self.load_batch(ix_range, dataset=dataset)
                    data = np.array(data)
                    labels = np.array(labels)
                    np.copyto(self._batch_dict[dataset][id]["data"], data)
                    np.copyto(self._batch_dict[dataset][id]["labels"], labels)
                    self._batch_dict[dataset][id]["start_ix"].value = start_ix
                    self._new_batch[dataset][id].value = 1
                    batches_aval = sum([self._new_batch[dataset][i].value for i in self._new_batch[dataset]])
                    if dataset == "train" and batches_aval < 5:
                        print("Prepared new batch, {} batches available".format(batches_aval))                    
                    self._locks[dataset][id].release()
            time.sleep(0.1)   
    
    def generate_batch(self, dataset="train"):
        """A generator reading data batches from shared memory.

        To be used by an external training function.
        
        Args:
            dataset: a dataset to use, either 'train' or 'val'.
        
        Yields:
            data: a numpy array containing data objects minibatch
            labels: a numpy array containing corresponding labels
        """
        while True:
            for id in self._new_batch[dataset]:
                if self._new_batch[dataset][id].value:
                    data = np.copy(self._batch_dict[dataset][id]["data"])
                    labels = np.copy(self._batch_dict[dataset][id]["labels"])
                    self._new_batch[dataset][id].value = 0
                    start_id = self._batch_dict[dataset][id]["start_ix"].value
                    if not start_id in self._unprocessed_batches_local[dataset]:
                        print("Warning, {} not in unprocessed".format(start_id))
                    else:
                        del self._unprocessed_batches_local[dataset][start_id]
                    yield data, labels

    def load_batch(self, ix, dataset='train'):
        """Loads a batch of images defined by indices in ix list.
        
        Args:
            dataset: a dataset to use, either 'train' or 'val'.
            ix: a list of integer ids to load, corresponding to indices
                in the `self._img_keys[dataset]` list.
        Returns:
              images: a list containing images corresponding to ids in ix.
              labels: a list containing corresponding labels.          
        """
        images = []
        labels = []
        for i in ix:
            filename = self._img_keys[dataset][i]
            outline = self._outlines[dataset][i]
            img, label = self.load_image(filename, outline, dataset)
            images.append(np.expand_dims(img, axis=-1))
            labels.append(np.expand_dims(label, axis=-1).astype("float32"))            
        return images, labels

    def preprocess_image(self, img, dataset):
        """Preprocesses a given image by e.g. resizing and scaling."""
        if not img.shape == self.img_dim:
            img = cv2.resize(img, self.img_dim, cv2.INTER_AREA)
        img = (img/255.0).astype('float32')
        if not (self._use_random_crops and dataset == "train"):
            return img
        shift = np.random.randint(0, self.crop_size, size=(2))
        new_shape = tuple(np.array(img.shape) + self.crop_size)
        new_img = np.zeros(new_shape, dtype="float32")
        half = self.crop_size//2
        new_img[half:-half, half:-half] = img
        cropped_img = new_img[
                      shift[0]:shift[0] + img.shape[0],
                      shift[1]:shift[1] + img.shape[1]
                      ]
        return cropped_img

    def load_image(self, filename, outline, dataset):
        """Loads and preprocesses a single image from a hard drive."""
        img = np.load(filename)
        img = self.preprocess_image(img, dataset)
        pil_mask = Image.new('L', img.shape, 0)
        outline_arr = [(point["x"], point["y"]) for point in outline]
        ImageDraw.Draw(pil_mask).polygon(outline_arr, outline=1, fill=1)  # draw a barcode mask
        label = np.array(pil_mask)
        return img, label


if __name__ == "__main__":
    d = BatchGen()
    for mb in d.generate_batch("train"):
        data, labels = mb
