#! /usr/bin/env python3
"""
A script to calibrate the PiCam module using a Charuco board
"""

import time
import json

import cv2
import numpy as np

from cv2 import aruco

from charuco import charucoBoard
from charuco import charucoDictionary
from charuco import detectorParams
from preview import show_fullscreen_image

import realsense as rs


NUM_VALID_THRESHOLD = 50
CAMERA_RESOLUTION = (1920, 1080)
SCREEN_RESOLUTION = (1920, 1080)
FRAME_SPACING = 5
CAMERA_CONFIG_PATH = 'etc/camera_config.json'


def save_json(data, filename=CAMERA_CONFIG_PATH):
    """
    Save our data object as json to the camera_config file
    :param data: data to  write to file
    """
    print('Saving to file: ' + filename)
    json_data = json.dumps(data)
    with open(filename, 'w') as jsonfile:
        jsonfile.write(json_data)


def load_camera_props(props_file=CAMERA_CONFIG_PATH):
    """
    Load the camera properties from file.  To build this file you need
    to run the aruco_calibration.py file
    :param props_file: Camera property file name
    """
    with open(props_file, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data.get('camera_matrix'))
    dist_coeffs = np.array(data.get('dist_coeffs'))
    return camera_matrix, dist_coeffs


def get_undistort_maps(camera_matrix, dist_coeffs):
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, CAMERA_RESOLUTION, 0)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, CAMERA_RESOLUTION, 5)
    return mapx, mapy


def undistort_image(image, mapx, mapy):
    """
    Given an image from the camera module, load the camera properties and correct
    for camera distortion
    :param image: Original, distorted image
    :param camera_matrix: Param from camera calibration
    :param dist_coeffs: Param from camera calibration
    :param prop_file: The camera calibration file
    :return: Corrected image
    """
    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return image


def get_charuco_corners(stream):
    """
    Given an VideoStream with a charuco calibration board in view,
    return a list of the charuco board corners and their corresponding
    ids
    """
    window_name = 'Calibrating...'
    corners = []
    ids = []
    frame_idx = 0

    # Read Images Loop
    while True:
        #cap_success, frame = stream.read()
        frame = stream.read_rgb()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, charucoDictionary, parameters=detectorParams)
        if marker_corners and frame_idx % FRAME_SPACING == 0:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, charucoBoard)
            if charuco_corners is not None and charuco_ids is not None:
                if len(charuco_corners) > 3:
                    corners.append(charuco_corners)
                    ids.append(charuco_ids)
                    print("Found: %d / %d" % (len(ids), NUM_VALID_THRESHOLD))
            aruco.drawDetectedMarkers(gray, marker_corners, marker_ids)

        cv2.imshow(window_name, gray)
        frame_idx += 1

        if cv2.waitKey(1) & 255 == ord('q'):
            break

        if len(ids) >= NUM_VALID_THRESHOLD:
            cv2.destroyWindow(window_name)
            return corners, ids


def calibrate_camera():
    """
    Calibrate our camera
    """
    # Display Charuco Calibration Board
    charuco = charucoBoard.draw(SCREEN_RESOLUTION)
    show_fullscreen_image(charuco)

    # Step 1: Initialize Camera
    #  stream = cv2.VideoCapture(0)
    #  stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #  stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    stream = rs.RealSenseCamera()
    time.sleep(2)

    # Step 2: Identify Charuco Corners and IDs
    corners, ids = get_charuco_corners(stream)

    # Step 3: Calculate and Save Camera Matrix & Distortion Coefficients
    print('Finished collecting data, computing...')
    try:
        err, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(corners, ids, charucoBoard, CAMERA_RESOLUTION, None, None)
        print('Calibrated with error: ', err)

        save_json({'camera_matrix': camera_matrix.tolist(),
                   'dist_coeffs': dist_coeffs.tolist(),
                   'err': err})

        print('...DONE')
    except Exception as e:
        print(e)

    # Step 4: Generate Undistortion / Rectify Map
    camera_matrix, dist_coeffs = load_camera_props(CAMERA_CONFIG_PATH)
    mapx, mapy = get_undistort_maps(camera_matrix, dist_coeffs)

    # Step 5: Show Calibrated Image
    while True:
        #cap_success, frame = stream.read()
        frame = stream.read_rgb()
        frame = undistort_image(frame, mapx, mapy)
        cv2.imshow('Calibrated Image', frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break

    # stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate_camera()
