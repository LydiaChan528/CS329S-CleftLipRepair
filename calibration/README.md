# Overview
Library used for calibration.

Based off of https://github.com/dclemmon/projection_mapping


# Usage
1. Run `calibrate.py` to get camera matrix coefficients.
2. Run `perspective.py` to isolate projectable area.

- You will want to run `calibrate.py` first to get the camera matrix in JSON form
- You will want to use `perspective.py` more as a library to return the
  transformation matrix `m` to isolate the projectable area
