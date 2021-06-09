import numpy as np
import keras
from keras.layers import *
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model
import time
import datetime
from keras.optimizers import Adam
import argparse



from barcode_predict.batch_gen import BatchGen
from keras.callbacks import ModelCheckpoint, TensorBoard


class KerasDataGenerator(keras.utils.Sequence):
    """"A generator class expected by a Keras model.

    Args:
        batch_gen: an object of a BatchGen class
        n_chanels: an integer number of channels in images (3 for RGB)
        shuffle: a bool whether to shuffle examples in the minibatch
        dataset: a dataset name to draw minibatches from ('train' or 'val')
    """
    def __init__(self, batch_gen, n_channels=1,
                 shuffle=True, dataset="train"):
        'Initialization'
        self.batch_gen = batch_gen
        self.dim = batch_gen.img_dim
        self.batch_size = batch_gen.batch_size[dataset]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.dataset = dataset
        self.epoch_proc_time = time.time()
        self.ix_count = np.zeros(self.batch_gen.data_size[dataset])

    def __len__(self):
        'Denotes the number of batches per epoch'
        batches_per_epoch = self.batch_gen.data_size[self.dataset] // self.batch_gen.batch_size[self.dataset]
        if self.dataset == "train":
            if self.batch_gen.data_size[self.dataset] > 5e5:
                batches_per_epoch //= 10    # define epoch as 1/10-th of a dataset size
            else:
                batches_per_epoch //= 2
        return batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        data, labels = next(self.batch_gen.generate_batch(dataset=self.dataset))
        labels = to_categorical(labels, num_classes=2)
        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return


class BarcodeModel:
    """A class representing a Keras model for recognizing barcodes in scanner images.

    Args:
        learning_rate: a float learning rate.
        epochs: an integer number of epochs to train for.
        aggregate_grads: a boolean whether to aggregate gradients across
            multiple epochs before applying them.
        gpu: an integer index of a GPU device to use.
        datadir: a path to a directory where the data is stored.
        architecture: a neural network architecture to use, e.g. "simple",
            "unet".
    """
    def __init__(self,
                 learning_rate=1e-4,
                 epochs=500,
                 aggregate_grads=True,
                 gpu=0,
                 datadir="/home/dmytro/Data/scanner_images",
                 architecture=None,
                 ):
        self.data_gen = BatchGen(
            datadir=datadir
        )
        self.keras_data_gen_train = KerasDataGenerator(
            batch_gen=self.data_gen,
            dataset="train"
        )
        self.keras_data_gen_val = KerasDataGenerator(
            batch_gen=self.data_gen,
            dataset="val"
        )
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


    def build_model_simple(self):
        """Builds the network symbolic graph in tensorflow."""
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        base_layers = [
            Conv2D(16, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(32, (3, 3), strides=(2, 2),
                                         activation="relu",
                                         padding='same'),

            Conv2D(32, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),

            Conv2D(64, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),

            Conv2D(64, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (3, 3), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (3, 3), strides=(1, 1),
                                             activation="relu",
                                             padding='same')
        ]
        decoder_layers = [
            Conv2DTranspose(64, (3, 3), strides=(2, 2),
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
            x = Conv2DTranspose(filters, (3, 3), strides=(2, 2),
                                activation="relu", padding="same")(x)
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

    def init_session(self):
        """Initializes tensorflow session."""
        config = tf.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.__enter__()

    def train(self):
        """Runs the model training loop."""
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        tensorboard_callback = TensorBoard(
            log_dir="logs/unet_3blocks_16filters_{}".format(current_time)
        )
        save_callback = ModelCheckpoint(
            filepath="saved_models/barcode_prediction_model_{}.pkl".format(current_time),
            save_weights_only=True,
            period=1
        )
        self.model.fit(x=self.keras_data_gen_train,
                       epochs=self.epochs,
                       validation_data=self.keras_data_gen_val,
                       workers=0,
                       verbose=2,
                       callbacks=[save_callback, tensorboard_callback]
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
        batches_processed = 0
        ious = []
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            iou = self.compute_iou(logits, labels)
            ious.append(iou)
            batches_processed += 1
            if batches_processed > self.data_gen.data_size['val']//self.data_gen.batch_size['val']:
                break
        print("mIOU", np.mean(np.hstack(ious)))





