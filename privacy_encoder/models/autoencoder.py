from tensorflow import keras

from .base import BaseAutoEncoder


class AutoEncoder(BaseAutoEncoder):
    def __init__(
        self,
        input_shape=(128, 128, 3),
        z_dim=64,
        encoder_weights_path=None,
        decoder_weights_path=None,
        autoencoder_weights_path=None,
    ):

        self.input_shape = input_shape
        self.n_channels = input_shape[-1]
        self.z_dim = z_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        if encoder_weights_path is not None:
            self.encoder.load_weights(encoder_weights_path)

        if decoder_weights_path is not None:
            self.decoder.load_weights(decoder_weights_path)

        self.model = self.build_autoencoder()

        if autoencoder_weights_path is not None:
            self.model.load_weights(autoencoder_weights_path)
            self.predict = self.predict_autoencoder
        else:
            self.predict = self.predict_encoder_decoder

        self.autoencoder = self.model
