import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(page_title="Lung Segmentation", page_icon="🫁", layout="wide")

# Title and description
st.title("🫁 Lung Segmentation from Chest X-rays")
st.markdown("""
Upload a chest X-ray image to get a segmentation mask of the lungs using our Dense UNet model.
""")

# Load the model (cache it to avoid reloading on every interaction)
@st.cache_resource
def load_segmentation_model():
    model = load_model('Dense_UNet_lung_segmentation.h5', compile=False)
    return model

model = load_segmentation_model()

# Preprocessing function (must match your training preprocessing)
def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess a single image exactly like during training
    Processes images in the same way as your original preprocess_data() function
    
    Args:
        image: Input image (file path, numpy array, or PIL Image)
        target_size: Target resolution (must match training size)
    
    Returns:
        Preprocessed image ready for model prediction (with batch dimension)
    """
    # Convert different input types to numpy array
    if isinstance(image, str):  # File path
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, Image.Image):  # PIL Image
        img = np.array(image)
        if len(img.shape) == 3:  # Convert RGB to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:  # Assume numpy array
        img = image.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. Resize to target dimensions
    img = cv2.resize(img, target_size)
    
    # 2. Apply CLAHE for contrast normalization (EXACTLY like training)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # 3. Convert to float32 and normalize to [0,1] range
    img = img.astype(np.float32) / 255.0
    
    # 4. Add channel dimension (now shape: (256,256,1))
    img = np.expand_dims(img, axis=-1)
    
    # 5. Add batch dimension (shape: (1,256,256,1))
    img = np.expand_dims(img, axis=0)
    
    return img

# Postprocessing function
def postprocess_mask(pred_mask, threshold=0.5):
    """
    Convert model output to a binary mask and resize to original dimensions
    """
    # Threshold predictions
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    
    # Remove batch and channel dimensions if present
    if len(binary_mask.shape) == 4:
        binary_mask = binary_mask[0]
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.squeeze()
    
    return binary_mask * 255  # Scale to 0-255 for display

# Function to overlay mask on image
def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay segmentation mask on the original image
    """
    # Ensure both are numpy arrays
    image = np.array(image)
    mask = np.array(mask)
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert mask to colored (red in this case)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color for mask
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlayed

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Preprocess
    processed_image = preprocess_image(image_np)
    
    # Predict
    with st.spinner("Segmenting lungs..."):
        pred_mask = model.predict(processed_image)
    
    # Postprocess
    binary_mask = postprocess_mask(pred_mask)
    
    # Create overlay
    overlay = overlay_mask(image_np, binary_mask)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(binary_mask, caption="Segmentation Mask", use_column_width=True, clamp=True)
    with col2:
        st.image(overlay, caption="Overlay on Original", use_column_width=True)
    
    # Add download buttons
    st.markdown("### Download Results")
    
    # Convert numpy arrays to PIL Images
    mask_img = Image.fromarray(binary_mask)
    overlay_img = Image.fromarray(overlay)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
        mask_img.save(tmp_mask.name)
        mask_path = tmp_mask.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_overlay:
        overlay_img.save(tmp_overlay.name)
        overlay_path = tmp_overlay.name
    
    # Create download buttons
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
    
    # Clean up temporary files
    os.unlink(mask_path)
    os.unlink(overlay_path)

# Add some info about the model
st.sidebar.markdown("""
### About the Model
This app uses a Dense UNet model trained on chest X-ray images for lung segmentation.

**Model Details:**
- Architecture: Dense UNet
- Input Size: 256×256 grayscale
- Training Data: [Montgomery County and Shenzhen datasets](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data)
""")