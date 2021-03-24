"""
Wrapper to work with Intel RealSense Camera
"""

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera():

    def __init__(self):
        # Intel RealSense Camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        #  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        #config.enable_stream(rs.stream.color, 1920, 1280, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def read_rgb(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_frame = np.asanyarray(color_frame.get_data())
            return color_frame
