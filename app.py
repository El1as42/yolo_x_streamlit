import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image


model = YOLO("last.pt")  # This single parameter determines which yolo version and size is used

st.title("YOLO11 Object Detection")
st.write("Model size: Large")
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    # Convert the picture from the camera input to a format usable by OpenCV
    image = Image.open(picture)
    # Convert to NumPy array (OpenCV format: BGR)
    image = np.array(image)  

    st.write("Running YOLO11 object detection...")
    try:
        # Perform inference on the input image
        results = model(image)  

        # Check if YOLO detected any objects
        if results and len(results[0].boxes) > 0:
            st.write(f"Detected {len(results[0].boxes)} objects.")
        else:
            st.write("No objects detected.")

        # Annotate the image with bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display the annotated image in Streamlit
        st.image(annotated_frame, use_column_width=True)
    except Exception as e:
        st.error(f"Error during object detection: {e}")
