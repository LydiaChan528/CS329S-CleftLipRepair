#! /usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model/'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'calibration/'))

import cv2
import numpy as np

import model.tools.infer as infer
import calibration.calibrate as calibrate
import calibration.perspective as perspective
import calibration.realsense as rs
from calibration.fullscreen.fullscreen import FullScreen

CAMERA_NUM = 0
PROP_FILE = 'calibration/etc/camera_config.json'
PERSPECTIVE_FILE = 'calibration/etc/perspective.json'
CAMERA_RESOLUTION = (1920, 1080)
QUIT_KEY = 'q'


def main():
    # Load Calibration Data
    camera_matrix, dist_coeffs = calibrate.load_camera_props(PROP_FILE)
    mapx, mapy = calibrate.get_undistort_maps(camera_matrix, dist_coeffs)
    m, max_width, max_height = perspective.load_perspective(PERSPECTIVE_FILE)

    # Configure Camera
    stream = rs.RealSenseCamera()
    #  stream = cv2.VideoCapture(CAMERA_NUM)
    #  stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    #  stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    # Configure Projector Display
    screen = FullScreen(0)

    i = 0
    WINDOW_LENGTH = 5
    pred_window = np.zeros((21, 2, WINDOW_LENGTH))
    while True:
        #  _, frame = stream.read()
        frame = stream.read_rgb()

        # Undistort
        frame = calibrate.undistort_image(frame, mapx, mapy)

        # Perspective Transform
        frame = cv2.warpPerspective(frame, m, (max_width, max_height))

        # Convert to RGB-order numpy array
        #  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)

        # Remove blue and reduce green channel to alleviate
        # self-interference of projected points
        frame[:, :, 2] = 0
        frame[:, :, 1:] = frame[:, :, 1:] // 2

        # Predict
        preds, face_det_img = infer.predict_single(frame, face_det=True)
        pred_window = np.append(pred_window, np.expand_dims(preds.numpy(), axis=-1), axis=2)

        pred_landmarks = np.mean(pred_window, axis=2)
        overlay, canvas = infer.plot_landmarks(face_det_img, pred_landmarks)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Display points only on Projector
        screen.imshow(canvas)

        # Display points overlayed on image on screen
        cv2.imshow("Model View", overlay)

        # Break on user pressing 'Q'
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break

        # Slide the window
        pred_window = pred_window[:, :, 1:]

    # Destroy OpenCV Structures
    cv2.destroyAllWindows()
    #  stream.release()


if __name__ == "__main__":
    main()
