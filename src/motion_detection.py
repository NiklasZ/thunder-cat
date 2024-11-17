import cv2 as cv
import numpy as np
from cv2 import BackgroundSubtractorMOG2
from cv2.typing import MatLike, Rect

from configuration import configure_logger

logger = configure_logger()


def detect_motion(
    frame: MatLike,
    subtractor: BackgroundSubtractorMOG2,
    draw_bounding_boxes: bool,
    minimum_area=1000,
) -> tuple[MatLike, list[Rect]]:

    labelled_frame = frame.copy()

    # Get foreground mask
    fg_mask = subtractor.apply(frame)

    # Variant A
    # Threshold the mask to remove shadows
    _, thresh = cv.threshold(fg_mask, 244, 255, cv.THRESH_BINARY)
    # Find contours in the thresholded mask
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected motion
    bounding_boxes = []

    if len(contours):
        areas = [cv.contourArea(c) for c in contours]
        idx = np.argmax(areas)
        largest_contour = contours[idx]
        if cv.contourArea(largest_contour) > minimum_area:  # Adjust this threshold to filter out noise
            rect = cv.boundingRect(largest_contour)
            bounding_boxes.append(rect)
            x, y, w, h = rect
            if draw_bounding_boxes:
                cv.rectangle(labelled_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return labelled_frame, bounding_boxes
