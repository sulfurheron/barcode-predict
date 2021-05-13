import numpy as np
import keras
from keras.layers import *
import tensorflow as tf
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.models import Model
import time
import datetime
from keras.applications import resnet50, densenet, nasnet, mobilenet_v2, xception
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.regularizers import l2
import argparse


from barcode_predict.data_gen import DataGen
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard


class KerasDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
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
                batches_per_epoch //= 10
            else:
                batches_per_epoch //= 2
        return batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print("epoch processing time", time.time() - self.epoch_proc_time)
        #start_time = time.time()
        data, labels = next(self.data_gen.generate_batch(dataset=self.dataset))
        labels = to_categorical(labels, num_classes=2)
        #print("epoch acquisition time", time.time() - start_time)

        #labels[labels == 0] = 0.01
        #labels[labels == 1] = 0.99
        # if self.dataset == "val":
        #     print("val counts", np.sum(self.ix_count == 1))
        # elif self.dataset == "train":
        #     print("train counts", np.sum(self.ix_count == 1))
        #print("batch index", index)
        #self.epoch_proc_time = time.time()
        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return


class LabelDistribution(keras.metrics.Metric):
    """
    Computes the per-batch label distribution (y_true) and stores the array as
    a metric which can be accessed via keras CallBack's

    :param n_class: int - number of distinct output class(es)
    """

    def __init__(self, **kwargs):
        super(LabelDistribution, self).__init__(**kwargs)


    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true = y_true
        self.y_pred = y_pred

    def result(self):
        return self.y_true, self.y_pred

    def reset_states(self):
        return


class ResnetModel:

    def __init__(self,
                 learning_rate=1e-4,
                 epochs=15,
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
        #with strategy.scope():
            self.init_session()
            if architecture == "xception":
                self.build_model_xception()
            else:
                self.build_model_simple()


    def build_model_simple(self):
        """Builds the network symbolic graph in tensorflow."""
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        base_layers = [
            Conv2D(16, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(32, (5, 5), strides=(2, 2),
                                         activation="relu",
                                         padding='same'),

            Conv2D(32, (5, 5), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),

            Conv2D(64, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),

            Conv2D(64, (5, 5), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (5, 5), strides=(1, 1),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (5, 5), strides=(2, 2),
                                             activation="relu",
                                             padding='same'),
            Conv2D(64, (5, 5), strides=(1, 1),
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

    def build_model_xception(self):
        """Builds the network symbolic graph in tensorflow."""
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        x = xception.Xception(
            weights='imagenet',
            input_shape=(self.input_shape),
            include_top=False,
            pooling="avg",
            #classes=2
        )(x)
        #self.output = x
        self.output = Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=self.img, outputs=self.output)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

    def build_model_impala(self):
        """Builds the network symbolic graph in tensorflow."""
        self.img = Input(name="input", shape=self.input_shape, dtype='float32')
        x = self.img
        channels = [16, 32, 32]
        for channel in channels:
            x = Conv2D(channel, (3, 3), strides=(1, 1),
                                             padding='same')(x)
            x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
            # residual blocks
            for j in range(2):
                block_input = x
                x = ReLU()(x)
                x = Conv2D(channel, (3, 3), strides=(1, 1),
                                                 activation='relu',
                                                 padding='same')(x)
                x = Conv2D(channel, (3, 3), strides=(1, 1),
                                                 padding='same')(x)
                x = Add()([x, block_input])
        x = ReLU()(x)
        x = GlobalMaxPooling2D()(x)
        self.output = Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=self.img, outputs=self.output)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

    def init_session(self):
        config = tf.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.__enter__()

    def train(self):
        self.all_losses = {'train': [], 'val': []}
        self.accs = {'val_sens': [], 'val_spec': [], 'val_sens_single': [], 'val_spec_single': []}
        # val_sens_single, val_spec_single = self.evaluate_on_validation_singles_set()
        # print("Validation singles accuracy: sensitivity {}, specificity {}".format(val_sens_single, val_spec_single))
        # self.evaluate_on_test_singles_set()
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
        tensorboard_callback = TensorBoard(log_dir="logs")
        save_callback = ModelCheckpoint(filepath="saved_models/barcode_prediction_model_{}.pkl".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                                        save_weights_only=True,
                                        period=1
                                        )
        self.model.fit(x=self.keras_data_gen_train,
                       epochs=self.epochs,
                       validation_data=self.keras_data_gen_val,
                       #steps_per_epoch=100 // self.data_gen.batch_size['train'],
                       #validation_steps=val_steps,
                       workers=0,
                       verbose=2,
                       callbacks=[
                           #on_epoch_end,
                           save_callback,
                           tensorboard_callback
                       ]
                       )

    def evaluate_on_validation_set(self):
        """Evaluate on orders combined from images from validation set."""
        #predictions = []
        class_labels = []
        predictions = []
        all_logits = []
        batches_processed = 0
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            #loss, acc = self.model.evaluate(data, to_categorical(labels, num_classes=self.num_classes), verbose=0)
            predictions.append(np.argmax(logits, axis=-1) == labels)
            class_labels.append(labels)
            all_logits.append(logits[:, 1])
            #preds.append(preds)
            batches_processed += 1
            if batches_processed > self.data_gen.data_size['val']//self.data_gen.batch_size['val']:
                break
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

    def build_roc(self, labels, scores):
        scores_sorted = sorted(scores)
        sens = []
        spec = []
        for i in range(len(scores_sorted)):
            pos_pred = scores >= scores_sorted[i]
            neg_pred = scores < scores_sorted[i]
            sens.append(np.sum((labels==1) * (pos_pred))/np.sum(labels==1))
            spec.append(1 - np.sum((labels==0) * neg_pred)/np.sum(labels==0))
        auc = np.mean(sens)
        return spec, sens, auc

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
        im_num = 1
        for mb in self.data_gen.generate_batch(dataset="val"):
            data, labels = mb
            logits = self.model.predict(data)
            # loss, acc = self.model.evaluate(data, to_categorical(labels, num_classes=self.num_classes), verbose=0)

            for i in range(len(data)):
                pred_error = np.linalg.norm(logits[i][:, :, 1:] - labels[i])
                if pred_error > 15:
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
                for i in range(im_num):
                    plt.subplot(im_num, 3, i * 3 + 1)
                    plt.imshow(imgs_to_show[i][:, :, 0], cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    plt.title("Image", fontsize=16)

                    plt.subplot(im_num, 3, i * 3 + 2)
                    plt.imshow(labels_to_show[i][:, :, 0], cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    plt.title("Label", fontsize=16)

                    plt.subplot(im_num, 3, i * 3 + 3)
                    plt.imshow(logits_to_show[i][:, :, 1], cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    plt.title("Prediction", fontsize=16)
                plt.show()

            imgs_to_show = []
            labels_to_show = []
            logits_to_show = []

    def show_pictures_no_scan(self):
        import pickle
        with open("ppo_vpred_images_prod_2.pkl", "rb") as f:
            data = pickle.load(f)
        for i in range(len(data['images'])):
            if np.random.random() > 1:
                continue
            print("i", i)
            # if not data['scanner_ids'][i] in ["1", "2"]:
            #     continue
            imgs = data['images'][i, :, :256*256*2]
            imgs = np.reshape(imgs, (2, 256, 256, 1)).astype('float32')/255.0
            logits = self.model.predict(imgs)
            logits[0, 0, 0, 1] = 1
            logits[0, 0, 1, 1] = 0
            logits[1, 0, 0, 1] = 1
            logits[1, 0, 1, 1] = 0
            plt.figure(figsize=(12, 6))

            imgs_to_show = imgs
            im_num = 1
            for j in range(im_num):
                plt.subplot(im_num, 2, j * 2 + 1)
                plt.imshow(imgs[j][:, :, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(im_num, 2, j * 2 + 2)
                plt.imshow(logits[j][:, :, 1], cmap='gray')
                plt.xticks([])
                plt.yticks([])
            plt.show()
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
    model = ResnetModel(
        epochs=args.epochs,
        datadir=args.datadir,
        gpu=args.gpu,
    )
    #model.load("saved_models/cnn_shared_5_layers_0.718869149684906_2021-03-01-09-51-29.pkl")
    #model.evaluate_on_validation_set()
    #model.show_pictures()
    #model.show_score_hist()
    #model.train()
    model.model.load_weights("saved_models/barcode_prediction_model_2021-05-12-18-19-15.pkl")
    model.show_pictures()
    #model.show_pictures_no_scan()
    #model.save()

