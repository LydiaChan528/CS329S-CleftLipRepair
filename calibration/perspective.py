#! /usr/bin/env python3
"""
This program calculates the perspective transform of the projectable area.
The user should be able to provide appropriate camera calibration information.
"""

import argparse
import json
import time

import cv2
import imutils
import numpy as np

import preview
import calibrate
import realsense as rs

PERSPECTIVE_PATH = 'etc/perspective.json'

def load_perspective(perspective_file=PERSPECTIVE_PATH):
    """
    Load the perspective transform from file
    """
    with open(perspective_file, 'r') as f:
        data = json.load(f)
    m = np.array(data.get('m'))
    max_width = np.array(data.get('max_width'))
    max_height = np.array(data.get('max_height'))
    return m, max_width, max_height


def get_reference_image(img_resolution=(1920, 1080)):
    """
    Build the image we will be searching for.  In this case, we just want a
    large blue box (full screen)
    :param img_resolution: this is our screen/projector resolution
    """
    width, height = img_resolution
    img = np.ones((height, width, 1), np.uint8) * 255
    return img


def find_edges(frame):
    """
    Given a frame, find the edges
    :param frame: Camera Image
    :return: Found edges in image
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Add some blur
    edged = cv2.Canny(gray, 30, 200)  # Find our edges
    return edged


def get_region_corners(frame):
    """
    Find the four corners of our projected region and return them in
    the proper order
    :param frame: Camera Image
    :return: Projection region rectangle
    """
    edged = find_edges(frame)
    # findContours is destructive, so send in a copy
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Sort our contours by area, and keep the 10 largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screen_contours = None

    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our contour has four points, we probably found the screen
        if len(approx) == 4:
            screen_contours = approx
            break
    else:
        print('Did not find contour')

    # Uncomment these lines to see the contours on the image
    cv2.drawContours(frame, [screen_contours], -1, (0, 255, 0), 3)
    cv2.imshow('Screen', frame)
    cv2.waitKey(0)
    pts = screen_contours.reshape(4, 2)
    rect = order_corners(pts)
    return rect


def order_corners(pts):
    """
    Given the four points found for our contour, order them into
    Top Left, Top Right, Bottom Right, Bottom Left
    This order is important for perspective transforms
    :param pts: Contour points to be ordered correctly
    """
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_destination_array(rect):
    """
    Given a rectangle return the destination array
    :param rect: array of points  in [top left, top right, bottom right, bottom left] format
    """
    (tl, tr, br, bl) = rect  # Unpack the values
    # Compute the new image width
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # Compute the new image height
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # Our new image width and height will be the largest of each
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    # Create our destination array to map to top-down view
    dst = np.array([
        [0, 0],  # Origin of the image, Top left
        [max_width - 1, 0],  # Top right point
        [max_width - 1, max_height - 1],  # Bottom right point
        [0, max_height - 1],  # Bottom left point
        ], dtype='float32')
    return dst, max_width, max_height


def get_perspective_transform(stream, screen_resolution, prop_file):
    """
    Determine the perspective transform for the current physical layout
    return the perspective transform, max_width, and max_height for the
    projected region
    :param stream: Video stream from our camera
    :param screen_resolution: Resolution of projector or screen
    :param prop_file: camera property file
    """
    reference_image = get_reference_image(screen_resolution)

    # Display the reference image
    preview.show_fullscreen_image(reference_image)
    cv2.waitKey(2000)

    # Grab a photo of the frame
    #cap_success, frame = stream.read()
    frame = stream.read_rgb()

    # Remove the reference image from the display
    cv2.destroyAllWindows()

    # We're going to work with a smaller image, so we need to save the scale
    ratio = frame.shape[0] / 300.0

    # Undistort the camera image
    camera_matrix, dist_coeffs = calibrate.load_camera_props()
    mapx, mapy = calibrate.get_undistort_maps(camera_matrix, dist_coeffs)
    frame = calibrate.undistort_image(frame, mapx, mapy)
    orig = frame.copy()

    # Resize our image smaller, this will make things a lot faster
    frame = imutils.resize(frame, height=300)

    rect = get_region_corners(frame)
    rect *= ratio  # We shrank the image, so now we have to scale our points up

    dst, max_width, max_height = get_destination_array(rect)

    m = cv2.getPerspectiveTransform(rect, dst)

    # Uncomment the lines below to see the transformed image
    #  warp = cv2.warpPerspective(orig, m, (max_width, max_height))
    #  cv2.imshow('Perspective Transform', warp)
    #  cv2.waitKey(0)

    return m, max_width, max_height


def parse_args():
    """
    A command line argument parser
    :return:
    """
    ap = argparse.ArgumentParser()
    # Camera frame resolution
    ap.add_argument('-cw', '--camera_width', type=int, default=1920,
                    help='Camera image width')
    ap.add_argument('-ch', '--camera_height', type=int, default=1088,
                    help='Camera image height')
    # camera property file
    ap.add_argument('-f', '--camera_props', default='etc/camera_config.json',
                    help='Camera property file')
    # Screen resolution
    ap.add_argument('-sw', '--screen_width', type=int, default=1920,
                    help='Projector or screen width')
    ap.add_argument('-sh', '--screen_height', type=int, default=1080,
                    help='Projector or screen height')
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # Camera frame resolution
    resolution = (args.get('camera_width'), args.get('camera_height'))

    #  stream = cv2.VideoCapture(0)
    #  stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #  stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    stream = rs.RealSenseCamera()
    time.sleep(2)  # Let the camera warm up

    prop_file = args.get('camera_props')
    camera_matrix, dist_coeffs = calibrate.load_camera_props(prop_file)
    mapx, mapy = calibrate.get_undistort_maps(camera_matrix, dist_coeffs)

    screen_res = (args.get('screen_width'), args.get('screen_height'))
    m, max_width, max_height = get_perspective_transform(stream, screen_res, args.get('camera_props'))

    calibrate.save_json({'m': m.tolist(), 'max_width': max_width,
                         'max_height': max_height}, filename=PERSPECTIVE_PATH)

    # Display a dark black image
    width, height = resolution
    img = np.zeros((height, width, 1), np.uint8)
    preview.show_fullscreen_image(img)

    while True:
        # Get an image
        #  cap_success, frame = stream.read()
        frame = stream.read_rgb()

        # Remove Distortion
        frame = calibrate.undistort_image(frame, mapx, mapy)

        # Perspective Transform
        frame = cv2.warpPerspective(frame, m, (max_width, max_height))

        cv2.imshow('Preview', frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cv2.destroyAllWindows()
