import os
import sys
import random
import streamlit as st

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

path = "../"
sys.path.append(path)

from privacy_encoder.models import AutoEncoder
from privacy_encoder.data import CelebA

MODEL_DIR = "../models/"
DATA_DIR = "../data/celeba/"
IMAGE_DIR = "../data/celeba/img_align_celeba_cropped/"
CLASSES = ["0", "1"]
FEATURES = "Eyeglasses"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
ENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stage1/encoder.h5")
DECODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stage1/decoder.h5")
AUTOENCODER_WEIGHTS_PATH = None

@st.cache(allow_output_mutation=True)
def get_random_neg_class():
    celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = celeba.split("test", drop_zero=False)
    paths = df[df[FEATURES] == 0]["image_id"].tolist()
    random.shuffle(paths)
    return paths


@st.cache(allow_output_mutation=True)
def get_random_pos_class():
    celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = celeba.split("test", drop_zero=False)
    paths = df[df[FEATURES] == 1]["image_id"].tolist()
    random.shuffle(paths)
    return paths


RANDOM_POS = get_random_pos_class()
RANDOM_NEG = get_random_neg_class()


params = {"INPUT_TYPE": "neg_class", "IMAGE_PATH": None, "BUTTON_PRESSED": False}


def get_params():

    # st.sidebar.title('Privacy Encoder')
    st.sidebar.title("Options")

    # Get image input
    options = ["negative_class", "positive_class"]
    params["INPUT_TYPE"] = st.sidebar.selectbox("Input Options", options)

    if params["INPUT_TYPE"] == "negative_class":
        x = random.choice(RANDOM_NEG)
        params["IMAGE_PATH"] = os.path.join(IMAGE_DIR, x)
    elif params["INPUT_TYPE"] == "positive_class":
        x = random.choice(RANDOM_POS)
        params["IMAGE_PATH"] = os.path.join(IMAGE_DIR, x)

    params["BUTTON_PRESSED"] = st.sidebar.button("Click here to plot")

    return params


def main(interface):

    params = get_params()

    # This keeps the order of what is displayed
    progress_bar = st.empty()
    image_loc = st.empty()
    image_original_loc = st.empty()

    if params["BUTTON_PRESSED"]:
        progress_bar.success("Loading Image...")
        params["IMAGE_ORIGINAL"] = interface.load_image(file_path=params["IMAGE_PATH"])
        progress_bar.success("Encoding & Decoding Image...")
        params["IMAGE"] = interface.predict(params["IMAGE_ORIGINAL"])
        progress_bar.success("Done!")
        image_loc.image(
            params["IMAGE"],
            use_column_width=False,
            width=360,
            channels="RGB",
            format="PNG",
        )
        image_original_loc.image(
            params["IMAGE_ORIGINAL"],
            use_column_width=False,
            width=360,
            channels="RGB",
            format="PNG",
        )


@st.cache(allow_output_mutation=True)
def setup():
    interface = AutoEncoder(
        input_shape=INPUT_SHAPE,
        z_dim=Z_DIM,
        encoder_weights_path=ENCODER_WEIGHTS_PATH,
        decoder_weights_path=DECODER_WEIGHTS_PATH,
        autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
    )
    return interface


if __name__ == "__main__":

    interface = setup()
    main(interface)
