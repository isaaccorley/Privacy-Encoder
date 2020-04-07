import os
import argparse
from pprint import pprint
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from privacy_encoder import models
from privacy_encoder.data import CelebA
from privacy_encoder.callbacks import ModelSaver

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


CLASSES = ["0", "1"]
FEATURES = "Eyeglasses"
N_CLASSES = 2

DATA_DIR = "/data/open-source/celeba/"
IMAGE_DIR = "img_align_celeba_cropped/"
INPUT_SHAPE = [128, 128, 3]
CLASSIFIER_WEIGHTS_PATH = None
EPOCHS = 10
BATCH_SIZE = 64

celeba = CelebA(image_folder=IMAGE_DIR, selected_features=[FEATURES])
celeba.attributes[FEATURES] = celeba.attributes[FEATURES].astype(str)
train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)

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

model = models.CelebAImageClassifier(
    input_shape=INPUT_SHAPE,
    n_classes=N_CLASSES,
    classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
)

pprint(model.model.input)
pprint(model.model.output)

### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.getcwd(), "logs/image_classifier/"),
    profile_batch=100000000
)
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/raw_image_classifier.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    "balanced", np.unique(train_generator.classes), train_generator.classes
)
print("Class Weights:\n", class_weights)

model.model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_split) // BATCH_SIZE,
    validation_steps=len(val_split) // BATCH_SIZE,
    callbacks=[reduce_lr, tb, early_stop, chkpnt],
    class_weight=class_weights,
    verbose=1,
    workers=8,
    max_queue_size=64,
)
