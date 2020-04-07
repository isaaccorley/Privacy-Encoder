from pprint import pprint
import matplotlib.pyplot as plt
import tensorflow as tf
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


CLASSES = ["0", "1"]
FEATURES = "Eyeglasses"
N_CLASSES = 2
DROP_ZERO = False

DATA_DIR = "./data/celeba/"
IMAGE_DIR = "img_align_celeba_cropped/"
INPUT_SHAPE = [128, 128, 3]
CLASSIFIER_WEIGHTS_PATH = "./models/base/raw_image_classifier.h5"
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
model = models.CelebAImageClassifier(
    input_shape=INPUT_SHAPE,
    n_classes=N_CLASSES,
    classifier_weights_path=CLASSIFIER_WEIGHTS_PATH
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
plt.title("Image Classifier: Train Set")
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
plt.title("Image Classifier: Validation Set")
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
plt.title("Image Classifier: Test Set")
plt.show()
