import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import imageio
import json
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import model_from_json

# --- ADD THESE FUNCTIONS AT THE TOP ---
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D((2,2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(256, 256, 3), base_filters=16):
    inputs = tf.keras.layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, base_filters)
    c2, p2 = encoder_block(p1, base_filters*2)
    c3, p3 = encoder_block(p2, base_filters*4)
    c4, p4 = encoder_block(p3, base_filters*8)
    b = conv_block(p4, base_filters*16)
    d4 = decoder_block(b, c4, base_filters*8)
    d3 = decoder_block(d4, c3, base_filters*4)
    d2 = decoder_block(d3, c2, base_filters*2)
    d1 = decoder_block(d2, c1, base_filters)
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(d1)
    return tf.keras.models.Model(inputs, outputs)

# Custom metrics with registration to enable loading
@register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


DRIVE_BASE = r"D:\Oi"
OUTPUT_DIR = os.path.join(DRIVE_BASE, "oil_spill_outputs")

# Load data
X_val = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(OUTPUT_DIR, "Y_val.npy"))

# Load model architecture and weights
# Create the model structure directly from the code
unet = build_unet(input_shape=(256, 256, 3), base_filters=16)
print("Model structure rebuilt successfully.")

# --- UPDATE THE WEIGHTS FILENAME ---
weights_path = os.path.join(OUTPUT_DIR, "unet_best_weights.weights.h5")

if os.path.exists(weights_path):
    unet.load_weights(weights_path)
    print("✅ Weights loaded successfully!")
else:
    print(f"❌ ERROR: Could not find {weights_path}")

# Compile with custom metrics for evaluation
unet.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy", iou_metric, dice_coefficient])

# Predict and threshold
preds = unet.predict(X_val, batch_size=8)
preds_binary = (preds > 0.5).astype(np.uint8)

def overlay_mask_on_image(image, mask, color=(1, 0, 0), alpha=0.5):
    img = np.clip(image, 0, 1)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    overlay = np.zeros_like(img)
    overlay[mask.astype(bool)] = color
    return img * (1 - alpha) + overlay * alpha

n_show = min(5, len(X_val))
idxs = np.random.choice(len(X_val), n_show, replace=False)

# Create output dirs
overlay_dir = os.path.join(OUTPUT_DIR, "example_overlays")
os.makedirs(overlay_dir, exist_ok=True)

# Visualize and save overlays
for i, idx in enumerate(idxs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X_val[idx])
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(Y_val[idx].squeeze(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_mask_on_image(X_val[idx], preds_binary[idx].squeeze(),
                                    color=(1, 0, 0), alpha=0.4))
    plt.title("Predicted Overlay")
    plt.axis("off")

    save_path = os.path.join(overlay_dir, f"validation_overlay_{i}.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Saved overlay image to {save_path}")

# Confusion matrix plotting and saving
cm = confusion_matrix(Y_val.flatten(), preds_binary.flatten())
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix (Pixel-wise)")
plt.xlabel("Predicted")
plt.ylabel("True")

conf_matrix_path = os.path.join(overlay_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()
print(f" Confusion matrix image saved to {conf_matrix_path}")

# Save overlays as separate PNG masks
for i, idx in enumerate(idxs[:3]):
    overlay_img = (overlay_mask_on_image(X_val[idx], preds_binary[idx].squeeze(),
                                        color=(1, 0, 0), alpha=0.4) * 255).astype(np.uint8)
    path = os.path.join(overlay_dir, f"overlay_mask_{i}.png")
    imageio.imwrite(path, overlay_img)
    print(f" Saved overlay mask image to {path}")

# Save training loss plot if history file exists
history_path = os.path.join(OUTPUT_DIR, "unet_history.json")
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure()
    plt.plot(history.get('loss', []), label='Training Loss')
    plt.plot(history.get('val_loss', []), label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    history_plot_path = os.path.join(OUTPUT_DIR, "training_history_loss.png")
    plt.savefig(history_plot_path)
    plt.close()
    print(f" Training history plot saved to {history_plot_path}")
else:
    print("Training history JSON not found; skipping loss plot saving.")
