import numpy as np
import keras
from keras.layers import *
import tensorflow as tf
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.metrics import Precision, MeanIoU, AUC
from keras.models import Model
import time
import datetime
from keras.applications import resnet50, densenet, nasnet, mobilenet_v2, xception
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.regularizers import l2
import argparse
import cv2


from barcode_predict.data_gen import DataGen
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard


class KerasDataGenerator(keras.utils.Sequence):
    "A generator class expected by a Keras model."
    def __init__(self, data_gen, n_channels=1,
                 shuffle=True, dataset="train"):
        'Initialization'
        self.data_gen = data_gen
        self.dim = data_gen.img_dim
        self.batch_size = data_gen.batch_size[dataset]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.dataset = dataset
        self.epoch_proc_time = time.time()
        self.ix_count = np.zeros(self.data_gen.data_size[dataset])

    def __len__(self):
        'Denotes the number of batches per epoch'
        batches_per_epoch = self.data_gen.data_size[self.dataset]//self.data_gen.batch_size[self.dataset]
        if self.dataset == "train":
            if self.data_gen.data_size[self.dataset] > 5e5:
                batches_per_epoch //= 1000    # define epoch as 1/10-th of a dataset size
            else:
                batches_per_epoch //= 2
        return batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        data, labels = next(self.data_gen.generate_batch(dataset=self.dataset))
        labels = to_categorical(labels, num_classes=2)
        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return


class BarcodeModel:
    """A class representing a Keras model for recognizing barcodes on scanner images."""
    def __init__(self,
                 learning_rate=1e-4,
                 epochs=500,
                 aggregate_grads=True,
                 gpu=0,
                 datadir="/home/dmytro/Data/scanner_images",
                 architecture=None,
                 ):
        self.data_gen = DataGen(
            datadir=datadir
        )
        self.keras_data_gen_train = KerasDataGenerator(data_gen=self.data_gen, dataset="train")
        self.keras_data_gen_val = KerasDataGenerator(data_gen=self.data_gen, dataset="val")
        self.input_shape = self.data_gen.full_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cumulative_grads = []
        self.grads_collected = 0
        self.aggregate_grads = aggregate_grads
        self.architecture = architecture
        mydevice = "/gpu:{}".format(gpu)
        with tf.device(mydevice):
            self.init_session()
            if architecture == "unet":
                self.build_model_unet()
            else:
                self.build_model_simple()

    def init_session(self):
        config = tf.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.__enter__()
        K.set_session(self.sess)

    def build_model_simple(self):
        """Builds the network symbolic graph in tensorflow."""
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        base_layers = [
            Conv2D(32, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (3, 3), strides=(2, 2),
                                         activation="relu",
                                         padding='same'),

            Conv2D(64, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),

            Conv2D(128, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),

            Conv2D(128, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(128, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(128, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(128, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            # Conv2D(64, (5, 5), strides=(1, 1),
            #                                  activation="relu",
            #                                  padding='same'),
            Conv2D(128, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same')
        ]
        decoder_layers = [
            Conv2DTranspose(128, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2DTranspose(64, (3, 3), strides=(2, 2),
                            activation="relu",
                            padding='same'),
            Conv2DTranspose(32, (3, 3), strides=(2, 2),
                            activation="relu",
                            padding='same'),
            Conv2DTranspose(32, (3, 3), strides=(2, 2),
                            activation="relu",
                            padding='same'),
            Conv2DTranspose(16, (3, 3), strides=(2, 2),
                            activation="relu",
                            padding='same')
        ]
        for layer in base_layers:
            x = layer(x)
        for layer in decoder_layers:
            x = layer(x)
        self.output = Conv2D(2, (3, 3), padding="same", activation="softmax")(x)
        self.model = Model(inputs=self.img, outputs=self.output)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

    def unet_conv_block(self, input, filters, kernel_size):
        """Builds a single u-net conv block."""
        x = Conv2D(filters, (kernel_size, kernel_size), strides=(1, 1),
                                             activation="relu",
                                             padding='same')(input)
        x = Conv2D(filters, (kernel_size, kernel_size), strides=(1, 1),
                   activation="relu",
                   padding='same')(x)
        return x

    def build_model_unet(self):
        """Builds a u-net model based on https://arxiv.org/abs/1505.04597"""
        filters = 16
        blocks = 3
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        contract_tensors = []
        for block in range(blocks):
            x = self.unet_conv_block(x, filters, 3)
            contract_tensors.append(x)
            x = MaxPool2D((2, 2))(x)
            filters *= 2

        for block in range(blocks):
            x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), activation="relu", padding="same")(x)
            x = Concatenate(axis=-1)([x, contract_tensors[-(block + 1)]])
            filters //= 2
            x = self.unet_conv_block(x, filters, 3)

        self.output = Conv2D(2, (1, 1), padding="same", activation="softmax")(x)
        self.model = Model(inputs=self.img, outputs=self.output)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

    # def init_session(self):
    #     """Initializes tensorflow session."""
    #     config = tf.ConfigProto(
    #         allow_soft_placement=True
    #     )
    #     config.gpu_options.allow_growth = True
    #     self.sess = tf.Session(config=config)
    #     self.sess.__enter__()

    def train(self):
        """Runs the model training loop."""
        self.all_losses = {'train': [], 'val': []}
        self.accs = {'val_sens': [], 'val_spec': [], 'val_sens_single': [], 'val_spec_single': []}
        self.max_acc = 0.71
        def print_logs(epoch, logs):
            #val_loss = self.evaluate_on_validation_set()
            #val_sens_single, val_spec_single = self.evaluate_on_validation_singles_set()
            #print("Loss: train {}, validation {}".format(logs['loss'], logs['val_loss']))
            # print("Validation accuracy: sensitivity {}, specificity {}".format(val_sens, val_spec))
            #print("Validation singles accuracy: sensitivity {}, specificity {}".format(val_sens_single, val_spec_single))
            print("val count", np.sum(self.keras_data_gen_val.ix_count == 1))
            self.keras_data_gen_val.ix_count.fill(0.0)
            self.all_losses['train'].append(logs['loss'])
            #self.all_losses['val'].append(logs['val_loss'])
            auc, acc, spec, sens = self.evaluate_on_validation_set()
            if auc > 0.8 and auc > self.max_acc:
                self.max_acc = auc
                self.save()
            #self.save()
        on_epoch_end = LambdaCallback(on_epoch_end=lambda epoch, logs: print_logs(epoch, logs))
        tensorboard_callback = TensorBoard(
            log_dir="logs/unet_3blocks_16filters_{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        )
        save_callback = ModelCheckpoint(
            filepath="saved_models/barcode_prediction_model_{}.pkl".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
            save_weights_only=True,
            period=1
        )
        self.model.fit(x=self.keras_data_gen_train,
                       epochs=self.epochs,
                       validation_data=self.keras_data_gen_val,
                       workers=0,
                       verbose=2,
                       callbacks=[
                           #on_epoch_end,
                           save_callback,
                           tensorboard_callback
                       ]
                       )

    def compute_iou(self, logits, labels):
        """Computes Intersection Over Union metrics for binary class case."""
        logits = np.argmax(logits, axis=-1)
        labels = labels[:, :, :, 0]
        inter = np.sum(logits * labels, axis=(1, 2))
        union = np.sum(logits, axis=(1, 2)) + np.sum(labels, axis=(1, 2)) - inter
        return inter/union


    def evaluate_on_validation_set(self):
        """Evaluates the model on a validation set."""
        #predictions = []
        class_labels = []
        predictions = []
        all_logits = []
        batches_processed = 0
        ious = []
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            iou = self.compute_iou(logits, labels)
            ious.append(iou)
            #loss, acc = self.model.evaluate(data, to_categorical(labels, num_classes=self.num_classes), verbose=0)
            #predictions.append(np.argmax(logits, axis=-1) == labels)
            #class_labels.append(labels)
            #all_logits.append(logits[:, 1])
            #preds.append(preds)
            batches_processed += 1
            if batches_processed > self.data_gen.data_size['val']//self.data_gen.batch_size['val']:
                break
        print("mIOU", np.mean(np.hstack(ious)))
        predictions = np.hstack(predictions)
        class_labels = np.hstack(class_labels)
        all_logits = np.hstack(all_logits)
        sens = np.sum(predictions[class_labels == 1]) / np.sum(class_labels == 1)
        spec = np.sum(predictions[class_labels == 0]) / np.sum(class_labels == 0)
        acc = np.mean(predictions)
        fpr, tpr, thresholds = roc_curve(class_labels, all_logits)
        auc = roc_auc_score(class_labels, all_logits)
        print("Validation acc, sens, spec, roc", acc, sens, spec, auc)
        cvb = 1
        return auc, acc, spec, sens
        #return np.mean(losses)


    def save(self):
        import pickle
        var_dict = {}
        for var in tf.global_variables():
            if 'model' in var.name:
                var_dict[var.name] = var.eval()
        with open("saved_models/cnn_shared_5_layers_{}_{}.pkl".format(self.max_acc, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), "wb") as f:
            pickle.dump({'weights': self.model.get_weights(),
                         "losses": self.all_losses,
                         "val_acc": self.accs}, f)

    def load(self, filename):
        import pickle
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)
        # assign_ops = []
        # for var in tf.global_variables():
        #     if 'model' in var.name:
        #         assign_ops.append(var.assign(model_dict['weights'][var.name]))
        # sess = tf.get_default_session()
        # sess.run(assign_ops)
        self.model.set_weights(model_dict['weights'])
        return model_dict

    def show_pictures(self):
        import matplotlib.pyplot as plt
        imgs_to_show = []
        labels_to_show = []
        logits_to_show = []
        im_num = 2
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            #logits[0, 0, 0, 1] = 1
            #logits[0, 0, 1, 1] = 0
            #logits[1, 0, 0, 1] = 1
            #logits[1, 0, 1, 1] = 0
            # loss, acc = self.model.evaluate(data, to_categorical(labels, num_classes=self.num_classes), verbose=0)

            for i in range(len(data)):
                pred_error = np.linalg.norm(logits[i][:, :, 1:] - labels[i])
                if pred_error > 20:
                    print("pred error", pred_error)
                    logits[i][0, 0, 1] = 0
                    logits[i][0, 1, 1] = 1
                    imgs_to_show.append(data[i])
                    labels_to_show.append(labels[i])
                    logits_to_show.append(logits[i])
                    if len(imgs_to_show) >= im_num:
                        break
            if len(imgs_to_show) >= im_num:
                plt.figure(figsize=(20, 12))
                if im_num > 2:
                    y_dim = 6
                    x_dim = im_num // 2
                else:
                    y_dim = 3
                    x_dim = im_num
                alpha = 0.6
                for i in range(im_num):
                    plt.subplot(x_dim, y_dim, (i // 2) * y_dim + (i%2) * 3 + 1)
                    gray_img = imgs_to_show[i]
                    #rgb_img = cv2.cvtColor((gray_img * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
                    rgb_img = (np.repeat(gray_img, repeats=3, axis=-1) * 255).astype('uint8')
                    heat_map = cv2.applyColorMap((logits_to_show[i][:, :, 1] * 255).astype("uint8"), cv2.COLORMAP_JET)
                    fin_img = cv2.addWeighted(rgb_img, alpha, heat_map, 1 - alpha, 0)
                    fin_img = fin_img[:, :, ::-1]
                    plt.imshow(fin_img)
                    plt.xticks([])
                    plt.yticks([])
                    if i < 2:
                        plt.title("Image", fontsize=16)

                    plt.subplot(x_dim, y_dim, (i // 2) * y_dim + (i%2) * 3 + 2)
                    plt.imshow(labels_to_show[i][:, :, 0], cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    if i < 2:
                        plt.title("Label", fontsize=16)

                    plt.subplot(x_dim, y_dim, (i // 2) * y_dim + (i%2) * 3 + 3)
                    plt.imshow(logits_to_show[i][:, :, 1], cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    if i < 2:
                        plt.title("Prediction", fontsize=16)

                plt.show()

                imgs_to_show = []
                labels_to_show = []
                logits_to_show = []

    def show_pictures_no_scan(self):
        import pickle
        # with open("ppo_vpred_images_prod_2.pkl", "rb") as f:
        #     data = pickle.load(f)
        data = np.load("rlscan_episodes_images.npy")
        plt.figure(figsize=(20, 12))
        imgs_to_show = []
        logits_to_show = []
        for i in range(len(data)):
            if np.random.random() > 1:
                continue
            print("i", i)
            # if not data['scanner_ids'][i] in ["1", "2"]:
            #     continue
            imgs = data[i:i+1].astype("float32")/255
            logits = self.model.predict(imgs)
            logits[0, 0, 0, 1] = 1
            logits[0, 0, 1, 1] = 0
            #logits[1, 0, 0, 1] = 1
            #logits[1, 0, 1, 1] = 0

            im_num = 4
            if im_num > 2:
                y_dim = 4
                x_dim = im_num // 2
            else:
                y_dim = 2
                x_dim = im_num
            if np.random.random() < 0.1:
                imgs_to_show.append(imgs)
                logits_to_show.append(logits)
            alpha = 0.6
            if len(imgs_to_show) >= im_num:
                for j in range(im_num):
                    plt.subplot(x_dim, y_dim, (j // 2) * y_dim + (j%2) * 2 + 1)
                    gray_img = imgs_to_show[j][0, :, :, :1]
                    # rgb_img = cv2.cvtColor((gray_img * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
                    rgb_img = (np.repeat(gray_img, repeats=3, axis=-1) * 255).astype('uint8')
                    heat_map = cv2.applyColorMap((logits_to_show[j][0, :, :, 1] * 255).astype("uint8"), cv2.COLORMAP_JET)
                    fin_img = cv2.addWeighted(rgb_img, alpha, heat_map, 1 - alpha, 0)
                    fin_img = fin_img[:, :, ::-1]

                    plt.imshow(fin_img)
                    plt.xticks([])
                    plt.yticks([])
                    if j < 2:
                        plt.title("Image", fontsize=16)
                    plt.subplot(x_dim, y_dim, (j // 2) * y_dim + (j%2) * 2 + 2)
                    plt.imshow(logits_to_show[j][0, :, :, 1], cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    if j < 2:
                        plt.title("Prediction", fontsize=16)
                plt.show()
                imgs_to_show = []
                logits_to_show = []
                plt.figure(figsize=(20, 12))

        cvb = 1

    def show_score_hist(self):
        all_labels = []
        all_logits = []
        batches_processed = 0
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            all_labels.append(labels)
            all_logits.append(logits[:, 1])
            batches_processed += 1
            if batches_processed > self.data_gen.data_size['val'] // self.data_gen.batch_size['val']:
                break
        all_labels = np.hstack(all_labels)
        all_logits = np.hstack(all_logits)
        fpr, tpr, thresholds = roc_curve(all_labels, all_logits)
        auc = roc_auc_score(all_labels, all_logits)
        plt.figure()
        plt.hist(all_logits[all_labels== 1], bins=20, alpha=1, label="scan success")
        plt.hist(all_logits[all_labels== 0], bins=20, alpha=0.7, label="scan fail")
        plt.legend()
        plt.grid()
        plt.title("Predicted score distribution")

        plt.figure()
        plt.plot(fpr, tpr,
                 label="AUC: {:.3f}".format(auc))
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RLScan offline analysis')
    parser.add_argument('--model', type=str,
                        default='resnet')  # 'lstm', 'padded_lstm', 'scapegoat', 'weighted_scapegoat' or 'comb_repr'
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--image_comb', type=str, default="time")
    parser.add_argument('--num_images', type=int, default=4)
    parser.add_argument('--color_mode', type=str, default="gray")
    args = parser.parse_args()
    model = BarcodeModel(
        epochs=args.epochs,
        datadir=args.datadir,
        gpu=args.gpu,
    )
    #model.model.load_weights("saved_models/barcode_prediction_model_2021-05-19-13-07-40.pkl")
    model.train()
    #model.evaluate_on_validation_set()
    #model.show_pictures()
    #model.show_pictures_no_scan()
    #model.save()

