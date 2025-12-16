import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_paste_button import paste_image_button
import pandas as pd
import tempfile
import os  # Added for file cleanup


# Privacy Notice:
# This application processes images and video frames in real time.
# No images or video streams are permanently stored.
# Only non-identifiable metadata is logged for reporting purposes.

# Page Setup
st.set_page_config(page_title="Safety Detection System", layout="wide")


# Fix Image Size (CSS)

st.markdown(
    """
    <style>
    img {
        max-height: 500px;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👷 Construction Site Safety Detection")
st.info("By using this application, you consent to real-time image processing. No images or videos are stored.")

# Load Model
@st.cache_resource
def load_model():
    # Ensure the path to your model is correct
    return YOLO('../models/best.pt') 

try:
    model = load_model()
    class_names = model.names
    st.sidebar.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


# 3. Sidebar (Control Panel)

st.sidebar.header("⚙️ Control Panel")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
st.sidebar.markdown("---")

# Privacy Options
st.sidebar.subheader("🛡️ Privacy & Ethics")
enable_privacy = st.sidebar.checkbox("Enable Privacy Blurring")
st.sidebar.markdown("---")

# Class Selection
st.sidebar.subheader("🎯 Select Classes")
if 'selected_options' not in st.session_state:
    st.session_state['selected_options'] = list(class_names.values())

def select_all_classes(): st.session_state['selected_options'] = list(class_names.values())
def clear_all_classes(): st.session_state['selected_options'] = []

btn_col1, btn_col2 = st.sidebar.columns(2)
with btn_col1: st.button("✅ Select All", on_click=select_all_classes, use_container_width=True)
with btn_col2: st.button("❌ Clear All", on_click=clear_all_classes, use_container_width=True)

selected_classes = st.sidebar.multiselect("Choose objects:", options=list(class_names.values()), key='selected_options')
selected_indices = [k for k, v in class_names.items() if v in selected_classes]


# 4. Unified Processing Function

def process_frame(frame_image):
    """
    Takes a PIL image, runs inference, handles privacy blurring, 
    and returns the plotted image and boxes.
    """
    # Prediction
    results = model.predict(frame_image, conf=confidence, classes=selected_indices)
    boxes = results[0].boxes
    
    # Convert PIL image to Numpy array for OpenCV processing
    img_array = np.array(frame_image)
    
    # Handle transparent PNGs (4 channels) -> Convert to RGB
    if img_array.shape[-1] == 4:
         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Privacy Processing (Blurring)
    if enable_privacy:
        # OpenCV uses BGR, PIL/Model uses RGB. Convert for processing.
        img_for_blur = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                # Check bounds and class name
                if cls_id in class_names and class_names[cls_id] == 'Person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Safe clipping (ensure coordinates are within image bounds)
                    h, w, _ = img_for_blur.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    roi = img_for_blur[y1:y2, x1:x2]
                    if roi.size > 0:
                        # Calculate kernel size based on box width
                        ksize = int((x2 - x1) / 5) | 1
                        if ksize < 3: ksize = 3
                        img_for_blur[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        
        # Convert back to RGB for display
        img_array = cv2.cvtColor(img_for_blur, cv2.COLOR_BGR2RGB)

    # Final Plotting (Draw boxes on the image)
    res_plotted = results[0].plot(img=img_array)
    return res_plotted, boxes


# 5. Main Application Interface (Tabs)

tab1, tab2, tab3 = st.tabs(["🖼️ Image Detection", "📷 Live Webcam", "🎥 Video File"])

#Tab 1: Image & Paste
with tab1:
    st.write("### Upload or Paste Image")
    col_input1, col_input2 = st.columns(2)
    img_input = None
    
    with col_input1:
        upl = st.file_uploader("Upload Image", type=['jpg','png', 'jpeg'])
        if upl: img_input = Image.open(upl)
            
    with col_input2:
        paste_res = paste_image_button("📋 Paste from Clipboard", background_color="#FF4B4B", text_color="#FFF")
        if paste_res.image_data is not None: img_input = paste_res.image_data

    if img_input:
        if st.button("🚀 Detect Image", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                res_img, boxes = process_frame(img_input)
                
                # Split display: Original vs Result
                st.write("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.write("**Original Image**")
                    st.image(img_input, use_container_width=True)
                
                with res_col2:
                    st.write("**Detection Result**")
                    st.image(res_img, use_container_width=True)
                
                # Report Generation
                if boxes:
                    data = [{"Object": class_names[int(b.cls[0])], "Conf": f"{float(b.conf[0]):.2f}"} for b in boxes]
                    df = pd.DataFrame(data)
                    st.success(f"Detected {len(df)} objects")
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Report (CSV)",
                        data=csv,
                        file_name="safety_report.csv",
                        mime="text/csv",
                    )
        else:
            # Preview
            st.image(img_input, caption="Preview", width=400)

# Tab 2: Webcam
with tab2:
    st.write("Click 'Take Photo' to capture a frame.")
    cam_input = st.camera_input("Camera")
    
    if cam_input:
        img_cam = Image.open(cam_input)
        res_img, boxes = process_frame(img_cam)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_cam, caption="Captured", use_container_width=True)
        with c2:
            st.image(res_img, caption="Result", use_container_width=True)

# Tab 3: Video File
with tab3:
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file:
        # Create a temp file to store the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(video_file.read())
        tfile_path = tfile.name
        tfile.close() # Close file handle so OpenCV can read it safely
        
        vf = cv2.VideoCapture(tfile_path)
        stframe = st.empty()
        
        if st.button("▶️ Start Video Analysis"):
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                res_plotted, _ = process_frame(pil_img)
                
                # Update the image placeholder
                stframe.image(res_plotted, caption="Processing Video...", width=700)
            
            vf.release()
            st.success("Video Processing Complete!")
            
            # Clean up: Delete the temp file
            try:
                os.unlink(tfile_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

st.sidebar.markdown("---")

st.sidebar.text("Safety AI System v1.0")
