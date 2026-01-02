import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from io import BytesIO

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


# --- Custom metrics ---
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

# --- Paths ---
DATASET_IMAGE_DIR = r"D:\Oi\oil_spill_dataset\train\images"
DATASET_MASK_DIR = r"D:\Oi\oil_spill_dataset\train\masks"
BASE_DIR = r"D:\Oi"
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "oil_spill_outputs", "unet_best_weights.weights.h5")
SAVE_DIR = os.path.join(BASE_DIR, "streamlit_user_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

# Updated to take ONLY the weights path
def load_model_local(weights_path):
    model = build_unet(input_shape=(256, 256, 3), base_filters=16)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric, dice_coefficient])
    return model

@st.cache_resource(show_spinner=True)
def get_model():
    # Only pass the weights path here
    return load_model_local(MODEL_WEIGHTS_PATH)

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4):
    image = np.array(image).astype(np.uint8)
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 1] = color
    return np.clip(image * (1 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)

def calculate_metrics(gt_mask, pred_mask, smooth=1):
    gt_f = gt_mask.flatten()
    pred_f = pred_mask.flatten()
    intersection = np.sum(gt_f * pred_f)
    dice = (2. * intersection + smooth) / (np.sum(gt_f) + np.sum(pred_f) + smooth)
    iou = (intersection + smooth) / (np.sum(gt_f) + np.sum(pred_f) - intersection + smooth)
    accuracy = np.sum(gt_f == pred_f) / len(gt_f)
    return dice, iou, accuracy

def plot_confusion(gt_mask, pred_mask):
    cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten())
    fig, ax = plt.subplots(figsize=(2.56, 2.56))  # Match ~256x256 pixels for display
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    return fig

def convert_image_to_bytes(img: Image.Image, ext="PNG"):
    buf = BytesIO()
    img.save(buf, format=ext)
    byte_im = buf.getvalue()
    return byte_im

@st.cache_resource(show_spinner=True)
def get_model():
    return load_model_local(MODEL_WEIGHTS_PATH)

model = get_model()

st.set_page_config(page_title="Oil Spill Segmentation", layout="wide")
st.title("🛢️ Oil Spill Segmentation with U-Net")

with st.sidebar:
    st.header("Input Options")
    dataset_images = sorted([f for f in os.listdir(DATASET_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    selected_img = st.selectbox("Select image from dataset:", ["-- Select --"] + dataset_images)
    uploaded_file = st.file_uploader("Or upload your own image (jpg/png):", type=["jpg", "jpeg", "png"])

img_pil = None
image_name = None

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    image_name = uploaded_file.name
elif selected_img != "-- Select --":
    img_path = os.path.join(DATASET_IMAGE_DIR, selected_img)
    img_pil = Image.open(img_path).convert("RGB")
    image_name = selected_img

if img_pil is not None:
    image_for_pred = img_pil.resize(IMG_SIZE)
else:
    image_for_pred = None

tabs = st.tabs(["Input Image", "Prediction", "Metrics"])

with tabs[0]:
    st.subheader("Input Image")
    if image_for_pred is not None:
        st.image(image_for_pred, use_container_width=False, width=256)
    else:
        st.info("Please upload or select an input image.")

with tabs[1]:
    st.subheader("Prediction")
    if image_for_pred is not None and st.button("Predict Oil Spill"):
        input_arr = np.array(image_for_pred) / 255.0
        with st.spinner("Running prediction..."):
            pred = model.predict(np.expand_dims(input_arr, axis=0))[0]
        pred_binary = (pred.squeeze() > 0.5).astype(np.uint8)
        input_save_path = os.path.join(SAVE_DIR, f"input_{image_name}")
        image_for_pred.save(input_save_path)
        st.sidebar.success(f"Saved input image: {input_save_path}")
        mask_save_path = os.path.join(SAVE_DIR, f"mask_{os.path.splitext(image_name)[0]}.png")
        Image.fromarray(pred_binary * 255).save(mask_save_path)
        st.sidebar.success(f"Saved predicted mask: {mask_save_path}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_for_pred, caption="Input (256×256)", width=256)
        with col2:
            st.image(pred_binary * 255, caption="Predicted Mask (256×256)", width=256)
        with col3:
            overlay_img = overlay_mask(np.array(image_for_pred), pred_binary, color=(255, 0, 0), alpha=0.4)
            st.image(overlay_img, caption="Overlay (256×256)", width=256)

        input_bytes = convert_image_to_bytes(image_for_pred)
        mask_bytes = convert_image_to_bytes(Image.fromarray(pred_binary * 255))
        overlay_bytes = convert_image_to_bytes(Image.fromarray(overlay_img))
        st.download_button("Download Input", input_bytes, file_name=f"input_{image_name}", mime="image/png")
        st.download_button("Download Mask", mask_bytes, file_name=f"mask_{os.path.splitext(image_name)[0]}.png", mime="image/png")
        st.download_button("Download Overlay", overlay_bytes, file_name=f"overlay_{os.path.splitext(image_name)[0]}.png", mime="image/png")

with tabs[2]:
    st.subheader("Metrics")
    if image_for_pred is None:
        st.info("Upload or select an image to see evaluation metrics.")
    else:
        if uploaded_file is None and selected_img != "-- Select --":
            gt_path = os.path.join(DATASET_MASK_DIR, os.path.splitext(selected_img)[0] + ".png")
            if os.path.exists(gt_path):
                gt_mask_img = Image.open(gt_path).convert("L").resize(IMG_SIZE)
                gt_mask = (np.array(gt_mask_img) > 127).astype(np.uint8)

                input_arr = np.array(image_for_pred) / 255.0
                pred = model.predict(np.expand_dims(input_arr, axis=0))[0]
                pred_binary = (pred.squeeze() > 0.5).astype(np.uint8)
                dice, iou, accuracy = calculate_metrics(gt_mask, pred_binary)
                st.metric("Dice Coefficient", f"{dice:.4f}")
                st.metric("IoU", f"{iou:.4f}")
                st.metric("Pixel Accuracy", f"{accuracy:.4f}")
                fig_cm = plot_confusion(gt_mask, pred_binary)
                st.pyplot(fig_cm, use_container_width=False)
            else:
                st.warning("Ground truth mask not found for selected dataset image.")
        else:
            st.info("Metrics display requires selecting a dataset image with a ground truth mask.")
