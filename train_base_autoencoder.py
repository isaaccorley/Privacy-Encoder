import os
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from privacy_encoder import models
from privacy_encoder.data import CelebA
from privacy_encoder.callbacks import Reconstruction

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

MODEL_DIR = "./models/base/"
DATA_DIR = "/data/open-source/celeba/"
IMAGE_DIR = "img_align_celeba_cropped/"
INPUT_SHAPE = [128, 128, 3]
Z_DIM = 512
ENCODER_WEIGHTS_PATH = None
DECODER_WEIGHTS_PATH = None
AUTOENCODER_WEIGHTS_PATH = None
EPOCHS = 10
BATCH_SIZE = 32


celeba = CelebA(image_folder=IMAGE_DIR)
train_split = celeba.split("training", drop_zero=False)
val_split = celeba.split("validation", drop_zero=False)
test_split = celeba.split("test", drop_zero=False)

datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=celeba.images_folder,
    x_col="image_id",
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="input",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_split,
    directory=celeba.images_folder,
    x_col="image_id",
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="input",
    shuffle=True,
    color_mode="rgb",
    interpolation="bilinear",
)

ae = models.AutoEncoder(
    input_shape=INPUT_SHAPE,
    z_dim=Z_DIM,
    encoder_weights_path=ENCODER_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
    autoencoder_weights_path=AUTOENCODER_WEIGHTS_PATH,
)
pprint(ae.model.input)
pprint(ae.model.output)

### Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
)
tb = keras.callbacks.TensorBoard(log_dir="./logs/base/autoencoder/")
chkpnt = keras.callbacks.ModelCheckpoint(
    filepath="./models/base/autoencoder.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
recon_sampler = Reconstruction(
    encoder=ae.encoder,
    decoder=ae.decoder,
    train_df=train_split,
    test_df=test_split,
    n_images=5,
    model_dir=MODEL_DIR,
    output_dir="./results/base/",
    image_dir=DATA_DIR + IMAGE_DIR,
)

ae.model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3), loss="mse", metrics=["mse", "mae"]
)

ae.model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_split) // BATCH_SIZE,
    validation_steps=len(val_split) // BATCH_SIZE,
    callbacks=[reduce_lr, tb, chkpnt, early_stop, recon_sampler],
    verbose=1,
    workers=8,
    max_queue_size=128,
)
