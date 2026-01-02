import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

DRIVE_BASE = r"D:\Oi"
OUTPUT_DIR = os.path.join(DRIVE_BASE, "oil_spill_outputs")
IMG_HEIGHT, IMG_WIDTH = 256, 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Load data
X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(OUTPUT_DIR, "Y_train.npy"))
X_val = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(OUTPUT_DIR, "Y_val.npy"))

def conv_block_local(x, filters, kernel_size=3, activation='relu', padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    return x

def encoder_block_local(x, filters):
    c = conv_block_local(x, filters)
    p = layers.MaxPooling2D((2,2))(c)
    return c, p

def decoder_block_local(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block_local(x, filters)
    return x

def iou_metric(y_true, y_pred, smooth=1):
    import tensorflow.keras.backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
iou_metric.__name__ = "iou_metric"

def dice_coefficient(y_true, y_pred, smooth=1):
    import tensorflow.keras.backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
dice_coefficient.__name__ = "dice_coefficient"

def build_hypermodel(hp):
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    base_filters = hp.Choice('base_filters', values=[16, 32])
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

    x = inputs
    c1, p1 = encoder_block_local(x, base_filters)
    c2, p2 = encoder_block_local(p1, base_filters * 2)
    c3, p3 = encoder_block_local(p2, base_filters * 4)
    c4, p4 = encoder_block_local(p3, base_filters * 8)

    b = conv_block_local(p4, base_filters * 16)
    b = layers.Dropout(dropout_rate)(b)

    d4 = decoder_block_local(b, c4, base_filters * 8)
    d3 = decoder_block_local(d4, c3, base_filters * 4)
    d2 = decoder_block_local(d3, c2, base_filters * 2)
    d1 = decoder_block_local(d2, c1, base_filters)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(d1)
    model = tf.keras.models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy", iou_metric, dice_coefficient])
    return model

tuner_dir = os.path.join(OUTPUT_DIR, "oil_spill_tuning")
weights_path = os.path.join(OUTPUT_DIR, "unet_tuner_best_weights.weights.h5")

tuner = kt.Hyperband(
    build_hypermodel,
    objective=kt.Objective("val_iou_metric", direction="max"),
    max_epochs=8,  # reduced for faster tuning
    factor=3,
    directory=tuner_dir,
    project_name='unet_hyperband'
)

earlystop_callback = EarlyStopping(monitor="val_iou_metric", mode="max", patience=4)

print("Starting tuner search (this may take a while).")
tuner.search(
    X_train, Y_train,
    epochs=8,
    validation_data=(X_val, Y_val),
    batch_size=8,
    callbacks=[earlystop_callback],
    verbose=1
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
print(f"base_filters: {best_hps.get('base_filters')}")
print(f"learning_rate: {best_hps.get('learning_rate')}")
print(f"dropout_rate: {best_hps.get('dropout_rate')}")

best_models = tuner.get_best_models(num_models=1)
if best_models:
    best_model = best_models[0]
    best_model.save_weights(weights_path)
    print(f"Saved best tuner model weights to {weights_path}")
else:
    print("No best model returned by tuner.")
