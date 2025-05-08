import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.models import load_model

st.set_page_config(page_title="Lung Segmentation", page_icon="🫁", layout="wide")
st.title("🫁 Lung Segmentation from Chest X-rays")
st.markdown("""
Upload a chest X-ray image to get a segmentation mask of the lungs using our Dense UNet model.
""")

@st.cache_resource
def load_segmentation_model():
    return load_model('Dense_UNet_lung_segmentation.h5', compile=False)

model = load_segmentation_model()

def preprocess_image(image, target_size=(256, 256)):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, Image.Image):
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = image.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, target_size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_mask(pred_mask, threshold=0.5):
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    if len(binary_mask.shape) == 4:
        binary_mask = binary_mask[0]
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.squeeze()
    return binary_mask * 255

def overlay_mask(image, mask, alpha=0.5):
    image = np.array(image)
    mask = np.array(mask).astype(np.uint8) # Ensure mask is uint8 for resizing

    # Resize the mask to the original image dimensions
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    colored_mask = np.zeros_like(image)
    colored_mask[resized_mask > 0] = [255, 0, 0]  # Red color for mask

    overlayed = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return overlayed

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_width, original_height = image.size # Get original dimensions
    # st.image(image, caption="Uploaded X-ray", use_column_width=True)
    image_np = np.array(image)
    processed_image = preprocess_image(image_np)

    with st.spinner("Segmenting lungs..."):
        pred_mask = model.predict(processed_image)

    binary_mask = postprocess_mask(pred_mask)

    # Resize the binary mask to the original image dimensions
    resized_binary_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    overlay = overlay_mask(image_np, resized_binary_mask)

    col_orig, col_mask, col_overlay = st.columns([1, 1, 1]) # Create three columns

    with col_orig:
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with col_mask:
        st.image(resized_binary_mask, caption="Segmentation Mask", use_column_width=True, clamp=True)

    with col_overlay:
        st.image(overlay, caption="Overlay on Original", use_column_width=True)

    st.markdown("### Download Results")

    mask_img = Image.fromarray(resized_binary_mask) # Use resized mask
    overlay_img = Image.fromarray(overlay)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
        mask_img.save(tmp_mask.name)
        mask_path = tmp_mask.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_overlay:
        overlay_img.save(tmp_overlay.name)
        overlay_path = tmp_overlay.name

    col1, col2 = st.columns(2)
    with col1:
        with open(mask_path, 'rb') as f:
            st.download_button(
                label="Download Mask",
                data=f,
                file_name="lung_mask.png",
                mime="image/png"
            )
    with col2:
        with open(overlay_path, 'rb') as f:
            st.download_button(
                label="Download Overlay",
                data=f,
                file_name="lung_segmentation_overlay.png",
                mime="image/png"
            )

    os.unlink(mask_path)
    os.unlink(overlay_path)

st.sidebar.markdown("""
### About the Model
This app uses a Dense UNet model trained on chest X-ray images for lung segmentation.

**Model Details:**
- Architecture: Dense UNet
- Input Size: 256×256 grayscale
- Training Data: [Montgomery County and Shenzhen datasets](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
""")