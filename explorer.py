"""
Data Explorer

Streamlit UI for browsing dataset images, annotations, and metadata.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model/'))

import glob
import streamlit as st
import pandas as pd
import torch
import numpy as np

import model.tools.infer as infer
import model.lib.core.function as function

CLEFT_IMAGES_DIR = "model/data/cleft/images/all/"
METADATA_PATH = "model/data/cleft/metadata/metadata_answers.csv"

st.title("Data Explorer")

image_paths = glob.glob(CLEFT_IMAGES_DIR + "**/*.jpg", recursive=True)

image_path = st.selectbox('Image to Display', image_paths)

df = pd.read_csv(METADATA_PATH)
image_num = int(os.path.splitext(os.path.split(image_path)[-1])[0])
df_row = df.loc[df['Image Number'] == image_num]
df_row = df_row.drop(df_row.columns[[6,7,8,9,10]], axis=1)

img = infer.get_cleft_image(image_path)
preds = torch.from_numpy(np.loadtxt(image_path[:-3] + "txt"))
preds = preds[:-2,].unsqueeze(0)
overlay, _ = infer.plot_landmarks(img, preds[0])

col1, col2 = st.beta_columns(2)
col1.header("Input Image")
col1.image(img / 255, use_column_width=True)

col2.header("Labeled Landmarks")
col2.image(overlay / 255, use_column_width=True)
st.text("Image Metadata")
st.table(df_row)

