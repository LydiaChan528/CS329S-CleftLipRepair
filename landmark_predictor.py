"""
Landmark Predictor

Streamlit UI for uploading or capturing a face image, running a trained model,
and  seeing the predicted landmarks overlaid on the image.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model/'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'calibration/'))

import glob
import cv2
import numpy as np
import streamlit as st

import model.tools.infer as infer

# Camera Identifier for OpenCV Video Capture
CAMERA_NUM = 2
CAMERA_RESOLUTION = (1920, 1080)

# In case drop down menu is used
CLEFT_IMAGES_DIR = "model/data/cleft/images/all/"


st.title("Landmark Predictor")
st.header("Inference on Uploaded Photos")

# File Uploader
upload = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])

# # Drop down menu
# image_paths = glob.glob(CLEFT_IMAGES_DIR + "**/*.jpg", recursive=True)
# upload = st.selectbox('Image to Display', image_paths)

if upload is not None:
    # Predict
    orig_img = infer.get_cleft_image(upload)
    preds, face_det_img = infer.predict_single(orig_img, face_det=True)
    overlay, canvas = infer.plot_landmarks(face_det_img, preds)

    # Display images
    col1, col2 = st.beta_columns(2)
    col1.header("Input Image")
    col1.image(orig_img / 255, use_column_width=True)
    col2.header("Predicted Landmarks")
    col2.image(overlay / 255, use_column_width=True)

st.markdown("---")


# Inference on Direct Camera Capture
st.header("Inference on Live Camera Capture")

# Placeholders for direct camera capture media
live_text = st.empty()
live_image = st.empty()

live = st.checkbox("Continuously Stream Video")
capture_camera = st.button("Capture From Camera")
if capture_camera:
    cap = cv2.VideoCapture(CAMERA_NUM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    while True:
        cap_success, cam_img = cap.read()

        if not cap_success:
            st.error('Failed to Capture from Camera')
            break

        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
        cam_img = np.array(cam_img)

        # Predict
        preds, face_det_img = infer.predict_single(cam_img, face_det=True)
        overlay, canvas = infer.plot_landmarks(face_det_img, preds)

        live_text.text("Live Stream: Input & Predicted Landmarks")
        live_image.image([cam_img / 255, overlay / 255], width=325)

        if not live:
            break

    cap.release()

st.markdown("---")


# Run application for real - with projection mapping
st.header("Surgery Time!")
live_project = st.button("Project Landmarks")
if live_project:
    subprocess.call("./project_landmarks.py", shell=False)
