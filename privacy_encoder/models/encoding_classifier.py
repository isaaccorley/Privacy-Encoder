import tensorflow_addons as tfa
from tensorflow import keras

from .base import BaseAutoEncoder


class EncodingClassifier(BaseAutoEncoder):
    def __init__(
        self,
        input_shape=(128, 128, 3),
        z_dim=64,
        n_classes=2,
        encoder_weights_path=None,
        classifier_weights_path=None,
        opt=keras.optimizers.Adam(lr=1e-3),
    ):

        self.input_shape = input_shape
        self.z_dim = z_dim
        self.n_classes = n_classes

        # Build autoencoder head
        self.encoder = self.build_encoder()

        if encoder_weights_path is not None:
            self.encoder.load_weights(encoder_weights_path)

        # Build classifier
        self.classifier = self.build_classifier()

        if classifier_weights_path is not None:
            self.classifier.load_weights(classifier_weights_path)

        self.model = self.build_combined(opt)

    def build_classifier(self):
        """ Build the encoding classifier """
        inputs = keras.layers.Input(shape=(self.z_dim,))
        x = keras.layers.Dense(units=32)(inputs)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dense(units=32)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dense(units=self.n_classes)(x)
        outputs = keras.layers.Activation("softmax")(x)
        model = keras.models.Model(inputs, outputs, name="Encoding_Classifier")
        print(model.summary())
        return model

    def build_combined(self, opt):
        """ Combine Encoder ->  Classifier """

        # Freeze the encoder
        for layer in self.encoder.layers:
            layer.trainable = False
        self.encoder.trainable = False

        inputs = keras.layers.Input(shape=self.input_shape)
        encoder_out = self.encoder(inputs)

        # Image Classifier output
        encoding_classifier_out = self.classifier(encoder_out)

        # Compile combined model
        model = keras.models.Model(
            inputs=inputs,
            outputs=encoding_classifier_out,
            name="EncodingClassifierWithEncoder",
        )

        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes=self.n_classes),
            ],
        )

        print(model.summary())
        return model