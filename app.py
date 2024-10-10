import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the YOLOv8 model (adjust to the appropriate path if needed)
model = YOLO("yolov8n.pt")  # Ensure you have the right path and model

# Streamlit app interface
st.title("YOLOv8 Object Detection")

enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    # Convert the picture from the camera input to a format usable by OpenCV
    image = Image.open(picture)
    image = np.array(image)  # Convert to NumPy array (OpenCV format: BGR)

    # Run YOLOv8 inference on the image
    st.write("Running YOLOv8 object detection...")

    try:
        results = model(image)  # Perform inference on the input image

        # Check if YOLO detected any objects
        if results and len(results[0].boxes) > 0:
            st.write(f"Detected {len(results[0].boxes)} objects.")
        else:
            st.write("No objects detected.")

        # Annotate the image with bounding boxes and labels
        annotated_frame = results[0].plot()  # YOLOv8 automatically annotates the image

        # Display the annotated image in Streamlit
        st.image(annotated_frame, caption="Detected objects with YOLOv8", use_column_width=True)
    except Exception as e:
        st.error(f"Error during YOLOv8 detection: {e}")
