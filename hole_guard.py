"""Check that calibrated delivery circles still match the live rectified image.

These broad hue checks are a safety interlock, not a gem classifier. If a circle
is occluded or its appearance changes, the operator must check the field.
"""
import cv2
import numpy as np

from auto_vision_gem_colors_v4_1 import COLOURS


HUE_RANGES = {
    'BLUE': (85, 120),
    'CYAN': (85, 120),
    'GREEN': (35, 85),
    'PURPLE': (125, 170),
    'ORANGE': (5, 30),
    'RED': (170, 10),  # wraps around OpenCV's 0..179 hue scale
}


def check_holes(frame, centres_cm, radii_cm):
    """Return labels that do not look like their assigned circle at the centre."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bad = []
    for colour, (cx, cy), radius in zip(COLOURS, centres_cm, radii_cm):
        x, y = round(cx * (width-1) / 210), round(cy * (height-1) / 120)
        r = max(5, min(12, round(radius * min((width-1)/210,
                                             (height-1)/120) / 3)))
        if x-r < 0 or x+r >= width or y-r < 0 or y+r >= height:
            bad.append(colour)
            continue
        patch = hsv[y-r:y+r+1, x-r:x+r+1]
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        patch = patch[xx*xx + yy*yy <= r*r]
        lo, hi = HUE_RANGES[colour]
        hue = patch[:, 0]
        match = ((hue >= lo) & (hue <= hi) if lo <= hi
                 else (hue >= lo) | (hue <= hi))
        match &= (patch[:, 1] >= 45) & (patch[:, 2] >= 45)
        if np.mean(match) < .60:
            bad.append(colour)
    return bad
