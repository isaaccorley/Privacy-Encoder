import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

from privacy_encoder.models import AutoEncoder
from privacy_encoder.data import CelebA

MODEL_DIR = "./models/"
DATA_DIR = "./data/celeba/"
IMAGE_DIR = "./data/celeba/img_align_celeba_cropped/"
RESULTS_DIR_RAW = "./output/raw"
RESULTS_DIR_RECON = "./output/reconstructed"
RESULTS_DIR_COMBINED = "./output/combined"
CLASSES = ["0", "1"]
FEATURES = "Eyeglasses"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
ENCODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stage1/encoder.h5")
DECODER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "privacy_stage1/decoder.h5")
AUTOENCODER_WEIGHTS_PATH = None


os.makedirs(RESULTS_DIR_RAW)
os.makedirs(RESULTS_DIR_RECON)
os.makedirs(RESULTS_DIR_COMBINED)

def get_neg_class():
    celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = celeba.split("test", drop_zero=False)
    paths = df[df[FEATURES] == 0]["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    random.shuffle(paths)
    return paths

def get_pos_class():
    celeba = CelebA(main_folder=DATA_DIR, selected_features=[FEATURES])
    df = celeba.split("test", drop_zero=False)
    paths = df[df[FEATURES] == 1]["image_id"].tolist()
    paths = [os.path.join(IMAGE_DIR, path) for path in paths]
    random.shuffle(paths)
    return paths

paths = get_pos_class()

privacy_encoder = AutoEncoder(
    input_shape=INPUT_SHAPE,
    z_dim=Z_DIM,
    encoder_weights_path=ENCODER_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
    autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
)

for path in tqdm(paths):
    image = privacy_encoder.load_image(file_path=path)
    recon = privacy_encoder.predict(image)
    combined = np.concatenate((image, recon), axis=1)

    image, recon, combined = Image.fromarray(image), Image.fromarray(recon), Image.fromarray(combined)
    image.save(os.path.join(RESULTS_DIR_RAW, os.path.basename(path)))
    recon.save(os.path.join(RESULTS_DIR_RECON, os.path.basename(path)))
    combined.save(os.path.join(RESULTS_DIR_COMBINED, os.path.basename(path)))
