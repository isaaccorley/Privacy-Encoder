import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K


class ModelSaver(keras.callbacks.Callback):
    def __init__(self, classifier, path):
        self.classifier = classifier
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
        self.classifier.save_weights(self.path)


class Reconstruction(keras.callbacks.Callback):
    def __init__(
        self,
        encoder,
        decoder,
        train_df,
        test_df,
        n_images,
        model_dir,
        output_dir,
        image_dir,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_images = n_images
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Get train set samples
        files = [os.path.join(image_dir, i) for i in train_df["image_id"].tolist()]
        random.shuffle(files)

        samples = random.sample(files, k=self.n_images)
        self.train_images = [self.load_image(sample) for sample in samples]

        # Get test set samples
        files = [os.path.join(image_dir, i) for i in test_df["image_id"].tolist()]
        random.shuffle(files)

        samples = random.sample(files, k=self.n_images)
        self.test_images = [self.load_image(sample) for sample in samples]

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def on_train_begin(self, logs={}):
        self.sample_images("init", self.train_images, set="train")
        self.sample_images("init", self.test_images, set="test")
        self.save_models()

    def on_epoch_end(self, epoch, logs={}):
        self.sample_images(epoch, self.train_images, set="train")
        self.sample_images(epoch, self.test_images, set="test")
        self.save_models()

    def save_models(self):
        self.encoder.save_weights(os.path.join(self.model_dir, "encoder.h5"))
        self.decoder.save_weights(os.path.join(self.model_dir, "decoder.h5"))

    def sample_images(self, epoch, images, set="train"):
        recon = [self.predict(image) for image in images]

        fig, axes = plt.subplots(self.n_images, 2)

        for i, row in enumerate(axes):
            row[0].imshow(images[i].squeeze().astype("uint8"))
            row[1].imshow(recon[i])
            row[0].axis("off")
            row[1].axis("off")

        plt.subplots_adjust(wspace=-0.7, hspace=0.05)
        fig.savefig(
            os.path.join(self.output_dir, "epoch_{}_{}.png".format(set, epoch)),
            bbox_inches="tight",
        )
        plt.close()

    def load_image(self, file_path):

        x = keras.preprocessing.image.load_img(
            file_path, grayscale=False, target_size=[128, 128, 3]
        )

        # Treat as PIL image
        x = keras.preprocessing.image.img_to_array(x, data_format=K.image_data_format())

        return x

    def predict(self, x):

        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        c = self.encoder.predict(x)
        y_pred = self.decoder.predict(c)
        y_pred = y_pred.squeeze()
        y_pred = y_pred * 255.0
        y_pred = np.clip(y_pred, a_min=0, a_max=255)
        y_pred = y_pred.astype("uint8")

        return y_pred


class ReconstructionByClass(Reconstruction):
    def __init__(
        self,
        encoder,
        decoder,
        neg_class_df,
        pos_class_df,
        n_images,
        model_dir,
        output_dir,
        image_dir,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_images = n_images
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Get train set samples
        files = [os.path.join(image_dir, i) for i in neg_class_df["image_id"].tolist()]
        #random.shuffle(files)

        #samples = random.sample(files, k=self.n_images)
        samples = files[:5]
        self.neg_images = [self.load_image(sample) for sample in samples]

        # Get test set samples
        files = [os.path.join(image_dir, i) for i in pos_class_df["image_id"].tolist()]
        #random.shuffle(files)

        #samples = random.sample(files, k=self.n_images)
        samples = files[:5]
        self.pos_images = [self.load_image(sample) for sample in samples]

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def on_train_begin(self, logs={}):
        self.sample_images("init", self.neg_images, set="neg")
        self.sample_images("init", self.pos_images, set="pos")
        self.save_models()

    def on_epoch_end(self, epoch, logs={}):
        self.sample_images(epoch, self.neg_images, set="neg")
        self.sample_images(epoch, self.pos_images, set="pos")
        self.save_models()
