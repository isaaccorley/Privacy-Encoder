import os
import glob
from tqdm import tqdm
from PIL import Image


DATA_DIR = "../data/celeba/"
IMAGE_DIR = "img_align_celeba"
CROP_DIR = "img_align_celeba_cropped"
NEW_WIDTH, NEW_HEIGHT = 128, 128

if not os.path.exists(os.path.join(DATA_DIR, CROP_DIR)):
    os.mkdir(os.path.join(DATA_DIR, CROP_DIR))

images = glob.glob(os.path.join(DATA_DIR, IMAGE_DIR, "*.jpg"))

for image in tqdm(images):
    im = Image.open(image)
    width, height = im.size

    left = (width - NEW_WIDTH)/2
    top = (height - NEW_HEIGHT)/2
    right = (width + NEW_WIDTH)/2
    bottom = (height + NEW_HEIGHT)/2

    im = im.crop((left, top, right, bottom))

    out_filename = image.replace(IMAGE_DIR, CROP_DIR)
    im.save(out_filename)