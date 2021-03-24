#! /usr/bin/env python3
"""
Utilities to display images in full screen and preview camera stream

Notes:
    - Change CAMERA_RESOLUTION to a lower resolution to improve
      frame rate
"""

import argparse

import cv2
import numpy as np

from charuco import charucoBoard
from fullscreen.fullscreen import FullScreen
import realsense as rs


SCREEN_RESOLUTION = (1920, 1080)
#  CAMERA_RESOLUTION = (1920, 1080)     # 1080p
#  CAMERA_RESOLUTION = (1280, 720)      # 720p
CAMERA_RESOLUTION = (640, 480)       # 480p
QUIT_KEY = 'q'


def show_fullscreen_image(frame):
    """
    Given an image, display the image in full screen.
    Use Case:
        > Display camera charuco pattern with projector.
        > Display camera preview
    """
    screen = FullScreen(0)
    screen.imshow(frame)


def preview_camera():
    """
    Display output of the camera
    """
    """
    # Traditional Webcam
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    """
    stream = rs.RealSenseCamera()
    while True:
        #_, frame = stream.read()
        frame = stream.read_rgb()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Camera Preview', frame)
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break


def preview_charuco():
    """
    Display an image until user presses "q" to quit
    """
    charuco = charucoBoard.draw(SCREEN_RESOLUTION)
    show_fullscreen_image(charuco)
    while True:
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break
    cv2.destroyAllWindows()


def preview_whiteboard(img_resolution=(1920, 1080)):
    """
    Display an all white image
    """
    width, height = img_resolution
    img = np.ones((height, width, 1), np.uint8) * 255
    show_fullscreen_image(img)
    while True:
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break
    cv2.destroyAllWindows()


def main():
    """
    Handle parsing of arguments
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--camera", action="store_true")
    group.add_argument("-b", "--board", action="store_true")
    group.add_argument("-w", "--whiteboard", action="store_true")
    args = parser.parse_args()
    if args.camera:
        preview_camera()
    elif args.board:
        preview_charuco()
    elif args.whiteboard:
        preview_whiteboard()
    else:
        preview_camera()


if __name__ == "__main__":
    main()
