import tensorflow_addons as tfa
from tensorflow import keras

from .base import BaseAutoEncoder
from .image_classifier import ImageClassifier
from .encoding_classifier import EncodingClassifier
from ..losses import weighted_categorical_crossentropy


class PrivacyAutoEncoder(BaseAutoEncoder):
    def __init__(
        self,
        input_shape=(128, 128, 3),
        z_dim=64,
        n_classes=2,
        encoder_weights_path=None,
        decoder_weights_path=None,
        autoencoder_weights_path=None,
        image_classifier_weights_path=None,
        encoding_classifier_weights_path=None,
        reconstruction_loss_weight=1.0,
        image_classifier_loss_weight=1.0,
        encoding_classifier_loss_weight=1.0,
        crossentropy_weights=[0.0, 1.0],
        opt=keras.optimizers.Adam(lr=1e-3),
    ):

        self.input_shape = input_shape
        self.n_channels = input_shape[-1]
        self.z_dim = z_dim
        self.n_classes = n_classes

        # Load autoencoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        if encoder_weights_path is not None:
            self.encoder.load_weights(encoder_weights_path)

        if decoder_weights_path is not None:
            self.decoder.load_weights(decoder_weights_path)

        self.autoencoder = self.build_autoencoder()

        if autoencoder_weights_path is not None:
            self.autoencoder.load_weights(autoencoder_weights_path)

        # Build classifiers
        self.image_classifier = self.load_image_classifier(
            image_classifier_weights_path
        )
        self.encoding_classifier = self.load_encoding_classifier(
            encoding_classifier_weights_path
        )

        # Construct combined model with frozen classifiers attached
        self.model = self.combine_models(
            opt,
            reconstruction_loss_weight,
            image_classifier_loss_weight,
            encoding_classifier_loss_weight,
            crossentropy_weights,
        )

    def load_image_classifier(self, classifier_weights_path):
        """ Load the image classifier and its pretrained weights """
        model = ImageClassifier(
            input_shape=self.input_shape,
            z_dim=self.z_dim,
            n_classes=self.n_classes,
            classifier_weights_path=classifier_weights_path,
        )
        model = model.classifier

        # Freeze the model
        for layer in model.layers:
            layer.trainable = False
        model.trainable = False

        return model

    def load_encoding_classifier(self, classifier_weights_path):
        """ Load the encoding classifier and its pretrained weights """
        model = EncodingClassifier(
            input_shape=self.input_shape,
            z_dim=self.z_dim,
            n_classes=self.n_classes,
            classifier_weights_path=classifier_weights_path,
        )
        model = model.classifier

        # Freeze the model
        for layer in model.layers:
            layer.trainable = False
        model.trainable = False

        return model

    def combine_models(
        self,
        opt,
        reconstruction_loss_weight,
        image_classifier_loss_weight,
        encoding_classifier_loss_weight,
        crossentropy_weights,
    ):
        """ Combine the autoencoder with the pretrained and encoding classifiers """

        inputs = keras.layers.Input(shape=self.input_shape)
        code = self.encoder(inputs)

        # Encoding classifier output
        encoding_classifier_out = self.encoding_classifier(code)
        encoding_classifier_out = keras.layers.Lambda(
            lambda x: x, name="encoding_classifier_output"
        )(encoding_classifier_out)

        # Freeze decoder
        for layer in self.decoder.layers:
            layer.trainable = False
        self.decoder.trainable = False

        # Decoder output
        decoder_out = self.decoder(code)
        decoder_out = keras.layers.Lambda(lambda x: x, name="decoder_output")(
            decoder_out
        )

        # Image Classifier output
        image_classifier_out = self.image_classifier(decoder_out)
        image_classifier_out = keras.layers.Lambda(
            lambda x: x, name="image_classifier_output"
        )(image_classifier_out)

        # Compile combined model
        model = keras.models.Model(
            inputs=inputs,
            outputs=[decoder_out, encoding_classifier_out, image_classifier_out],
            name="PrivacyAutoencoderWithClassifiers",
        )

        model.compile(
            opt,
            loss={
                "decoder_output": "mse",
                "encoding_classifier_output": weighted_categorical_crossentropy(
                    crossentropy_weights
                ),
                "image_classifier_output": weighted_categorical_crossentropy(
                    crossentropy_weights
                ),
            },
            loss_weights={
                "decoder_output": reconstruction_loss_weight,
                "encoding_classifier_output": encoding_classifier_loss_weight,
                "image_classifier_output": image_classifier_loss_weight,
            },
            metrics={
                "decoder_output": ["mse", "mae"],
                "encoding_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
                "image_classifier_output": [
                    "accuracy",
                    keras.metrics.AUC(),
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    tfa.metrics.F1Score(num_classes=self.n_classes),
                ],
            },
        )

        print(model.summary())
        return model
