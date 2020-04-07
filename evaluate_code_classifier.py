import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from privacy_encoder import models
from privacy_encoder.data import CelebA

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


'''
parser = argparse.ArgumentParser()
parser.add_argument('--encoder_dir', type=str, default="privacy_stage1", help='Directory containing encoder weights')
parser.add_argument('--classifier_dir', type=str, default="base", help='Directory containing classifier weights')
parser.add_argument('--classifier_dir', type=str, default="base", help='Directory containing classifier weights')
args = parser.parse_args()
'''


CLASSES = ["0", "1"]
FEATURES = "Eyeglasses"
N_CLASSES = 2
DROP_ZERO = False

ENCODER_DIR = "privacy_stage1" # args.encoder_dir
CLASSIFIER_DIR = "base" # args.classifier_dir
CLASSIFIER_TYPE = "image"
DATA_DIR = "./data/celeba/"
IMAGE_DIR = "img_align_celeba_cropped/"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
ENCODER_WEIGHTS_PATH = "./models/{}/encoder.h5".format(ENCODER_DIR)
DECODER_WEIGHTS_PATH = "./models/{}/decoder.h5".format(ENCODER_DIR)
CLASSIFIER_WEIGHTS_PATH = "./models/{}/{}_classifier.h5".format(CLASSIFIER_DIR, CLASSIFIER_TYPE)
BATCH_SIZE = 256

celeba = CelebA(image_folder=IMAGE_DIR, selected_features=[FEATURES])
celeba.attributes[FEATURES] = celeba.attributes[FEATURES].astype(str)
train_split = celeba.split("training", drop_zero=DROP_ZERO)
val_split = celeba.split("validation", drop_zero=DROP_ZERO)
test_split = celeba.split("test", drop_zero=DROP_ZERO)

#%% Construct datagens
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
    shuffle=False,
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
    shuffle=False,
    color_mode="rgb",
    interpolation="bilinear",
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_split,
    directory=celeba.images_folder,
    x_col="image_id",
    y_col=FEATURES,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    shuffle=False,
    color_mode="rgb",
    interpolation="bilinear",
)

#%% Define model
if CLASSIFIER_TYPE == "image":
    model = models.ImageClassifier(
        input_shape=INPUT_SHAPE,
        z_dim=Z_DIM,
        n_classes=N_CLASSES,
        encoder_weights_path=ENCODER_WEIGHTS_PATH,
        decoder_weights_path=DECODER_WEIGHTS_PATH,
        autoencoder_weights_path=None,
        classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
    )
else:
    model = models.EncodingClassifier(
        input_shape=INPUT_SHAPE,
        z_dim=Z_DIM,
        n_classes=N_CLASSES,
        encoder_weights_path=ENCODER_WEIGHTS_PATH,
        classifier_weights_path=CLASSIFIER_WEIGHTS_PATH,
    )

pprint(model.model.input)
pprint(model.model.output)


#%% Evaluate train set
y_pred = model.model.predict(
    train_generator,
    verbose=1,
    workers=2,
    max_queue_size=64,
)

y_pred = y_pred.argmax(axis=-1)
y_true = train_split[FEATURES].astype(int).values

# Metrics
report = classification_report(y_true, y_pred)
print(report)
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
class_names = ["No Eyeglasses", "Eyeglasses"]
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names)
plt.title("{} Classifier: Train Set".format(CLASSIFIER_TYPE.capitalize()))
plt.show()

#%% Evaluate val set
y_pred = model.model.predict(
    val_generator,
    verbose=1,
    workers=2,
    max_queue_size=64,
)

y_pred = y_pred.argmax(axis=-1)
y_true = val_split[FEATURES].astype(int).values

# Metrics
report = classification_report(y_true, y_pred)
print(report)
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
class_names = ["No Eyeglasses", "Eyeglasses"]
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names)
plt.title("{} Classifier: Validation Set".format(CLASSIFIER_TYPE.capitalize()))
plt.show()

#%% Evaluate test set
y_pred = model.model.predict(
    test_generator,
    verbose=1,
    workers=2,
    max_queue_size=64,
)

y_pred = y_pred.argmax(axis=-1)
y_true = test_split[FEATURES].astype(int).values

# Metrics
report = classification_report(y_true, y_pred)
print(report)
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
class_names = ["No Eyeglasses", "Eyeglasses"]
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names)
plt.title("{} Classifier: Test Set".format(CLASSIFIER_TYPE.capitalize()))
plt.show()
