"""
Projections Comparator

Compare the projections/predictions of two different models

By default: the pretrained model and the trained-from-scratch model
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model/'))

import glob
import streamlit as st
import numpy as np

from PIL import Image

DISPLAY_IMAGES_DIR = "./model/output/cleft/face_alignment_cleft_hrnet_w18/output_images/"
CLEFT_IMAGES_DIR = "./model/data/cleft/images/all/"
W18_IMAGES_DIR = "./model/output/cleft/face_alignment_cleft_hrnet_w18/output_images/projections/"
#SCR_IMAGES_DIR = "./model/output/cleft/train_nopretrained/output_images/projections/"
SCR_IMAGES_DIR = "./model/output/cleft/temp/output_images/projections/"

def load_image(image_path, blue=False):
  image = Image.open(image_path).convert('RGB')
  if blue: 
    image = np.array(image)
    image[:, :, [0,1,2]] = image[:, :, [2,1,0]]
    image = Image.fromarray(image, 'RGB')
  return image

st.title("Compare Projections")

image_paths = glob.glob(DISPLAY_IMAGES_DIR + "/*.jpg", recursive=True)
image_path = st.selectbox('Image to Predict On', image_paths)

datapoint = image_path.split("/")[-1]
preds = "proj_" + datapoint
img = load_image(CLEFT_IMAGES_DIR + datapoint)

preds_18 = load_image(W18_IMAGES_DIR + preds)
preds_scr = load_image(SCR_IMAGES_DIR + preds, blue=True)

overlay_w18 = Image.blend(img, preds_18, 0.25)
overlay_scr = Image.blend(img, preds_scr, 0.25)
overlay_all = Image.blend(overlay_scr, preds_18, 0.25)

img = np.array(img, dtype=np.float32)
overlay_w18 = np.array(overlay_w18, dtype=np.float32)
overlay_scr = np.array(overlay_scr, dtype=np.float32)
overlay_all = np.array(overlay_all, dtype=np.float32)

st.text("Input Image")
st.image(img / 255)
st.text("Predicted Landmarks from W18 Model")
st.image(overlay_w18 / 255)
st.text("Predicted Landmarks from Scratch Model")
st.image(overlay_scr / 255)
st.text("All Predicted Landmarks")
st.image(overlay_all / 255)
