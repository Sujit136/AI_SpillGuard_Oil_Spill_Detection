# model_define_train.py

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DRIVE_BASE = r"D:\Oi"
OUTPUT_DIR = os.path.join(DRIVE_BASE, "oil_spill_outputs")
IMG_HEIGHT, IMG_WIDTH = 256, 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(OUTPUT_DIR, "Y_train.npy"))
X_val = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(OUTPUT_DIR, "Y_val.npy"))

print("Loaded training and validation data.")

def conv_block(x, filters, kernel_size=3, activation='relu', padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=INPUT_SHAPE, num_classes=1, base_filters=64):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, base_filters)
    c2, p2 = encoder_block(p1, base_filters*2)
    c3, p3 = encoder_block(p2, base_filters*4)
    c4, p4 = encoder_block(p3, base_filters*8)
    b = conv_block(p4, base_filters*16)
    d4 = decoder_block(b, c4, base_filters*8)
    d3 = decoder_block(d4, c3, base_filters*4)
    d2 = decoder_block(d3, c2, base_filters*2)
    d1 = decoder_block(d2, c1, base_filters)
    outputs = layers.Conv2D(1, (1,1), activation='sigmoid', dtype='float32')(d1)
    model = models.Model(inputs, outputs)
    return model

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
iou_metric.__name__ = "iou_metric"

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
dice_coefficient.__name__ = "dice_coefficient"

unet = build_unet(input_shape=INPUT_SHAPE, base_filters=16)
unet.compile(optimizer="adam", loss="binary_crossentropy",
             metrics=["accuracy", iou_metric, dice_coefficient])
unet.summary()

# Updated weights filename with required extension '.weights.h5'
weights_path = os.path.join(OUTPUT_DIR, "unet_best_weights.weights.h5") 

checkpoint = ModelCheckpoint(weights_path, monitor='val_iou_metric', mode='max',
                             save_best_only=True, save_weights_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_iou_metric', mode='max', patience=8,
                          restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                              min_lr=1e-6, verbose=1)

BATCH_SIZE = 8
EPOCHS = 50

history = unet.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

hist_path = os.path.join(OUTPUT_DIR, "unet_history.json")
with open(hist_path, "w") as f:
    json.dump({k: [float(x) for x in v] for k,v in history.history.items()}, f)
print("Training complete. Weights saved to:", weights_path)
print("History saved to:", hist_path)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(OUTPUT_DIR, "training_history_loss.png")
plt.savefig(loss_plot_path)
plt.close()
print(f" Training history loss plot saved to {loss_plot_path}")
