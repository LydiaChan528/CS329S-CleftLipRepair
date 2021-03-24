"""
Constants and definitions for charuco board used in calibration
"""

from cv2 import aruco

inToM = 0.0254

# Camera calibration info
maxWidthIn = 17
maxHeightIn = 23
maxWidthM = maxWidthIn * inToM
maxHeightM = maxHeightIn * inToM

charucoNSqVert = 10
charucoSqSizeM = float(maxHeightM) / float(charucoNSqVert)
charucoMarkerSizeM = charucoSqSizeM * 0.7
charucoNSqHoriz = 16

charucoDictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
charucoBoard = aruco.CharucoBoard_create(
    charucoNSqHoriz,
    charucoNSqVert,
    charucoSqSizeM,
    charucoMarkerSizeM,
    charucoDictionary)

perspectiveDictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# We only want four corners for the perspective transform
perspectiveBoard = aruco.CharucoBoard_create(3, 3, charucoSqSizeM,
                                             charucoMarkerSizeM,
                                             perspectiveDictionary)

markerDictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
markerSizeIn = 5
markerSizeM = markerSizeIn * inToM

detectorParams = aruco.DetectorParameters_create()
