import os
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from privacy_encoder import models
from privacy_encoder.data import CelebA, multioutput_datagen
from privacy_encoder.callbacks import ReconstructionByClass

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


parser = argparse.ArgumentParser()
parser.add_argument('--nextstage', type=int, help='Next Stage')
parser.add_argument('--prevstage', type=int, help='Previous Stage')
args = parser.parse_args()

NEW = "privacy_stage{}".format(args.nextstage)
PREV = "privacy_stage{}".format(args.prevstage)

CLASSES = ["0", "1"]
FEATURES = "Eyeglasses"
N_CLASSES = 2

MODEL_DIR = "./models/{}/".format(NEW)
DATA_DIR = "/data/open-source/celeba/"
IMAGE_DIR = "img_align_celeba_cropped/"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
ENCODER_WEIGHTS_PATH = "./models/{}/encoder.h5".format(PREV)
DECODER_WEIGHTS_PATH = "./models/{}/decoder.h5".format(PREV)
AUTOENCODER_WEIGHTS_PATH = None
IMAGE_CLASSIFIER_WEIGHTS_PATH = "./models/{}/image_classifier.h5".format(PREV)
ENCODING_CLASSIFIER_WEIGHTS_PATH = "./models/{}/encoding_classifier.h5".format(PREV)
EPOCHS = 2
BATCH_SIZE = 128

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

celeba = CelebA(image_folder=IMAGE_DIR, selected_features=[FEATURES])

# Flip labels to force autoencoder to fool classifiers
celeba.attributes[FEATURES] = celeba.attributes[FEATURES].replace({0: 1, 1: 0})

celeba.attributes[FEATURES] = celeba.attributes[FEATURES].astype(str)

train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)
test_split = celeba.split("test", drop_zero=False)

# Random oversample smaller class (labels are flipped so glasses == '0')
high = len(train_split[train_split[FEATURES] != "0"][FEATURES])
low = len(train_split[train_split[FEATURES] == "0"][FEATURES])
n_samples = high - low
oversamples = train_split[train_split[FEATURES] == "0"].sample(n_samples, replace=True)
train_split = pd.concat([train_split, oversamples])

# Create datagens
datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=celeba.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=celeba.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

# Build model
ae = models.PrivacyAutoEncoder(
    input_shape=INPUT_SHAPE,
    z_dim=Z_DIM,
    n_classes=N_CLASSES,
    encoder_weights_path=ENCODER_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
    autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
    image_classifier_weights_path=IMAGE_CLASSIFIER_WEIGHTS_PATH,
    encoding_classifier_weights_path=ENCODING_CLASSIFIER_WEIGHTS_PATH,
    reconstruction_loss_weight=1.0,
    image_classifier_loss_weight=1e-2,
    encoding_classifier_loss_weight=1e-2,
    crossentropy_weights=[1.0, 1e-5],
    opt=keras.optimizers.Adam(lr=1e-3),
)
pprint(ae.model.input)
pprint(ae.model.output)


### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(log_dir="./logs/{}/privacy_autoencoder/".format(NEW))
'''
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/privacy_autoencoder_with_classifiers.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
'''
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Reconstruction callback
neg_class_df = test_split[test_split[FEATURES] != "0"]
pos_class_df = test_split[test_split[FEATURES] == "0"]
recon_sampler = ReconstructionByClass(
    encoder=ae.encoder,
    decoder=ae.decoder,
    neg_class_df=neg_class_df,
    pos_class_df=pos_class_df,
    n_images=5,
    model_dir=MODEL_DIR,
    output_dir="./results/{}/".format(NEW),
    image_dir=DATA_DIR + IMAGE_DIR,
)

# Train
ae.model.fit(
    multioutput_datagen(train_generator),
    validation_data=multioutput_datagen(val_generator),
    epochs=EPOCHS,
    steps_per_epoch=len(train_split) // BATCH_SIZE,
    validation_steps=len(val_split) // BATCH_SIZE,
    callbacks=[reduce_lr, tb, early_stop, recon_sampler],
    verbose=1,
    workers=1,
    max_queue_size=256,
)
