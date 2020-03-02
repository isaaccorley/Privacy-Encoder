import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


class BaseAutoEncoder(object):
    def build_encoder(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"
        )(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"
        )(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"
        )(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=self.z_dim)(x)
        outputs = keras.layers.Activation("relu")(x)
        model = keras.models.Model(inputs, outputs, name="Encoder")
        print(model.summary())
        return model

    def build_decoder(self):
        start_size = 16
        start_filters = 64
        inputs = keras.layers.Input(shape=(self.z_dim,))
        x = keras.layers.Dense(units=start_size * start_size * start_filters)(inputs)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Reshape((start_size, start_size, start_filters))(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Conv2D(
            filters=self.n_channels, kernel_size=(3, 3), padding="same"
        )(x)
        model = keras.models.Model(inputs, outputs, name="Decoder")
        print(model.summary())
        return model

    def build_autoencoder(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        code = self.encoder(inputs)
        outputs = self.decoder(code)
        model = keras.models.Model(inputs, outputs, name="AutoEncoder")
        print(model.summary())
        return model

    def load_image(self, file_path):

        x = keras.preprocessing.image.load_img(
            file_path, grayscale=False, target_size=self.input_shape[:2]
        )

        # Treat as PIL image
        x = keras.preprocessing.image.img_to_array(x, data_format=K.image_data_format())

        x = x.squeeze().astype("uint8")

        return x

    def predict_autoencoder(self, x):

        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        y_pred = self.model.predict(x)
        y_pred = y_pred.squeeze()
        y_pred = y_pred * 255.0
        y_pred = np.clip(y_pred, a_min=0, a_max=255)
        y_pred = y_pred.astype("uint8")

        return y_pred

    def predict_encoder_decoder(self, x):

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
