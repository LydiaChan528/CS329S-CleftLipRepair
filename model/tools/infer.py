"""
Inference API for cleft lip model
- For inferencing on a single given image
- To run on the pre-selected test split, use test.py instead:
    python tools/test.py --cfg experiments/cleft/face_alignment_cleft_hrnet_w18.yaml --model-file output/cleft/face_alignment_cleft_hrnet_w18/final_state.pth
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

import lib.models as models
from lib.utils import transforms
from lib.config import config, update_config
from lib.core.evaluation import decode_preds


# Paths from the root directory
MODEL_FILE = 'model/output/cleft/face_alignment_cleft_hrnet_w18/final_state.pth'
CONFIG_FILE = 'model/experiments/cleft/face_alignment_cleft_hrnet_w18.yaml'
LANDMARK_LABELS = ['lala', 'rala', 'lsbal', 'rsbbal', 'lc', 'rc', 'sn',
                   'lcphs', 'rcphs', 'rcphi', 'mcphi', 'lcphi', 'ls', 'sto',
                   'lch', 'rch', 'prn', 'rlr(r)', 'rlr(p)', 'rla', 'cp']
# Face detection model
CC_DIR = "model/tools/haarcascade_frontalface_default.xml"

# For normalizing the pixel values
PIX_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIX_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.determinstic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED

config.defrost()
config.merge_from_file(CONFIG_FILE)
config.freeze()

# Load landmark prediction model
model = models.get_face_alignment_net(config)

# Check GPU Available
gpus = list(config.GPUS)
if torch.cuda.is_available():
    print("PyTorch CUDA Available")

    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Special state_dict ordering if using CUDA
    state_dict = torch.load(MODEL_FILE)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)

    # Instantiate face detector
    print("Instantiating face detector...")
    detector = MTCNN(device='cuda')
else:
    state_dict = torch.load(MODEL_FILE, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    print("Instantiating face detector...")
    detector = MTCNN()


def get_cleft_image(unformatted_img):
    """Converts image to correct format

    Parameters
    ----------
    unformatted_img : str || Streamlit UploadedFile object

    Returns
    -------
    out_img : np array
    """
    out_img = np.array(Image.open(unformatted_img).convert('RGB'), dtype=np.float32)
    return out_img


def __preprocess_img(img, out_w, out_h, face_det=False, verbose=False):
    """Preprocesses image before inferencing

    Parameters
    ----------
    img : np array
        The input image in np array format, as returned by get_cleft_image()
    out_w, out_h : int
        Desired width and height of output image
    face_det : bool
        If true, runs MTCNN face detection and sets the output image
        to a crop of the input image that just includes the face

    Returns
    -------
    out_img :
        The image to run inferencing on, a crop if a face was detected
    face_det_img : np array
        If face detected, input image with bounding box
        Else, original input image
    scale, center : float
        Scale and center values of out_img
    """
    out_img = img.copy()
    face_det_img = img.copy()
    in_w = img.shape[1]
    in_h = img.shape[0]

    scale = None
    center = None
    face_found = False # switches to True if face detection succeeds
    if face_det:
        if verbose: print("Doing face detection...")
        #faces = detector.detect_faces(img)
        faces = detector.detect(img)
        color = (0, 255, 0)
        if len(faces) >= 1 and faces[0] is not None:
            face_found = True
            #x, y, w, h = faces[0]['box']
            x, y, x2, y2 = faces[0][0]
            w = x2 - x
            h = y2 - y

            # Drawing face bounding box
            cv2.rectangle(face_det_img, (x, y), (x + w, y + h), color, 2)

            # Crop to the face only
            scale = w * 1.0 / 200
            center = torch.tensor([x + w / 2.0, y + h / 2.0])
            out_img = transforms.crop(out_img, center, scale, [out_h, out_w], 0)
        else:
            if verbose: print("No faces found!")

    # Even if face detection fails, still try to predict
    if not face_found:
        scale = in_w * 1.0 / 200
        center = torch.tensor([in_w / 2.0, in_h / 2.0])
        out_img = transforms.crop(out_img, center, scale, [out_h, out_w], 0)

    out_img = np.array(out_img, dtype=np.float32)
    out_img = (out_img/255.0 - PIX_MEAN) / PIX_STD
    out_img = out_img.transpose([2, 0, 1])
    return out_img, face_det_img, scale, center


def predict_single(img, face_det=False):
    """Predicts landmarks on input image

    Parameters
    ----------
    img : np array
        The input image
    face_det : bool
        If true, will run face detection to crop the face before prediction

    Returns
    -------
    single_preds : list
        List of predicted (x,y) landmark points
    """
    img_size = config.MODEL.IMAGE_SIZE

    img, face_det_img, scale, center = __preprocess_img( img, img_size[1], img_size[0], face_det)

    img = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        score_map = output.data.cpu()
        preds = decode_preds(score_map, [center], [scale], score_map.shape[2:4])
        single_preds = preds[0] # batch size is 1, just take first
        return single_preds, face_det_img


def plot_landmarks(img, preds):
    """Plots predicted landmarks on image

    Parameters
    ----------
    img : np array
        The input image
    preds : list of (x,y)
        Predicted landmark locations

    Returns
    -------
    overlay : np array
        Original image with the predictions overlaid
    canvas : np array
        White image with the predictions
    """
    overlay = img.copy()
    # canvas is meant to be projected out
    canvas = np.zeros(img.shape, dtype=np.uint8)

    for i, point in enumerate(preds):
        x, y = point
        # Display Predictions on Original Image
        cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 0), -1)
        cv2.putText(overlay, LANDMARK_LABELS[i], (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

        # Display Predictions on white Canvas
        cv2.circle(canvas, (int(x), int(y)), 1, (255, 0, 0), 0)
        #  cv2.putText(canvas, LANDMARK_LABELS[i], (int(x) + 5, int(y) - 5),
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (127, 127, 127))

    return overlay, canvas
