"""The frame a line crop was cut in (crops.csv from make_crops.py), for scripts that compare crops with face coordinates.

A crop's rectangle x0..y1 is in the original face rotated about one point. If line_angle_deg is filled, that is the
line's own frame (rotated by line_angle_deg about line_cx, line_cy); otherwise it is the deskewed face (rotated by skew_deg
about the face centre).
"""
import cv2
import numpy as np


def crop_matrix(r, W, H):
    """2 x 3 matrix from original face pixels to the frame of crop row r, as cv2.warpAffine takes it. W, H: face size."""
    if r.get('line_angle_deg') not in ('', None):
        return cv2.getRotationMatrix2D((float(r['line_cx']), float(r['line_cy'])), float(r['line_angle_deg']), 1.0)
    return cv2.getRotationMatrix2D((W / 2, H / 2), float(r['skew_deg']), 1.0)


def box_in_frame(M, x0, y0, x1, y1):
    """Axis-aligned bounds (x0, y0, x1, y1) of a face-pixel box after matrix M."""
    P = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], float) @ np.asarray(M).T
    return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()


def line_row(r):
    """Row of the line's centre in the frame of crop row r. The rotation of a tilted crop turns about that point."""
    return float(r['line_cy']) if r.get('line_angle_deg') not in ('', None) else float(r['y_center'])
