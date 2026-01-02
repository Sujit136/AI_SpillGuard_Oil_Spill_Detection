# setup_and_load.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Updated paths for your D: drive
DRIVE_BASE = r"D:\Oi"
DATASET_BASE = os.path.join(DRIVE_BASE, "oil_spill_dataset")
IMAGE_DIR = os.path.join(DATASET_BASE, "train", "images")
MASK_DIR  = os.path.join(DATASET_BASE, "train", "masks")
OUTPUT_DIR = os.path.join(DRIVE_BASE, "oil_spill_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 256, 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

print("TensorFlow version:", tf.__version__)
print("IMAGE_DIR:", IMAGE_DIR)
print("MASK_DIR:", MASK_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("INPUT_SHAPE:", INPUT_SHAPE)

# Verify dataset presence
image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
mask_paths  = sorted(glob(os.path.join(MASK_DIR, "*.png")))

print("Found images:", len(image_paths))
print("Found masks:", len(mask_paths))

if len(image_paths) == 0 or len(mask_paths) == 0:
    raise FileNotFoundError("No images or masks found in specified directories.")

def basename_no_ext(p): return os.path.splitext(os.path.basename(p))[0]
common = set(basename_no_ext(p) for p in image_paths) & set(basename_no_ext(p) for p in mask_paths)
print("Matched filenames:", len(common))

# Load sample image and mask visualization
sample_img = image_paths[0]
sample_mask = os.path.join(MASK_DIR, basename_no_ext(sample_img) + ".png")
print("Sample image:", sample_img)
print("Sample mask:", sample_mask)

img_pil = load_img(sample_img)
mask_pil = load_img(sample_mask)
img_arr = img_to_array(img_pil).astype(np.uint8)
mask_arr = img_to_array(mask_pil).astype(np.uint8)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_arr.astype('uint8'))
plt.title("Sample Image")
plt.axis('off')

plt.subplot(1,2,2)
if mask_arr.ndim==3 and mask_arr.shape[2] > 1:
    plt.imshow(mask_arr[:,:,0], cmap='gray')
else:
    plt.imshow(mask_arr, cmap='gray')
plt.title("Sample Mask")
plt.axis('off')
plt.show()

print("Sample image shape:", img_arr.shape, "dtype:", img_arr.dtype)
print("Sample mask shape:", mask_arr.shape, "dtype:", mask_arr.dtype)

def load_dataset(image_dir, mask_dir, img_h=IMG_HEIGHT, img_w=IMG_WIDTH):
    images = sorted(glob(os.path.join(image_dir, "*.jpg")))
    masks  = sorted(glob(os.path.join(mask_dir, "*.png")))

    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}

    X, Y = [], []
    for img_path in tqdm(images, desc="Loading images"):
        key = os.path.splitext(os.path.basename(img_path))[0]
        if key not in mask_map:
            continue
        mask_path = mask_map[key]
        img = load_img(img_path, target_size=(img_h, img_w), color_mode="rgb")
        mask = load_img(mask_path, target_size=(img_h, img_w), color_mode="grayscale")
        img = img_to_array(img).astype(np.float32) / 255.0
        mask = img_to_array(mask).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        if img.shape[-1] != 3:
            img = np.repeat(img[..., :1], 3, axis=-1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, -1)
        elif mask.shape[-1] > 1:
            mask = mask[..., :1]
        X.append(img)
        Y.append(mask)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X, Y = load_dataset(IMAGE_DIR, MASK_DIR)
print("Loaded X shape:", X.shape, "Y shape:", Y.shape)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED)
print("Train set:", X_train.shape, Y_train.shape)
print("Val set:", X_val.shape, Y_val.shape)

# Save numpy arrays to OUTPUT_DIR
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val)
print("Saved numpy arrays to", OUTPUT_DIR)