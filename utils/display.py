import random
import colorsys

import cv2
import numpy as np


def get_color(tag, hue_step=0.41):
    tag = int(tag)
    h, v = (tag*hue_step) % 1, 1. - (int(tag*hue_step)%4)/5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(r*255), int(255*g), int(255*b)

def get_color_mask(mask, color=(0, 255, 0)):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask >= 200] = color[0]
    g[mask >= 200] = color[1]
    b[mask >= 200] = color[2]
    return np.stack([r, g, b], axis=2)

def draw_mask(frame, box, mask):
    canvas = np.zeros_like(frame).astype(np.uint8)
    xmin, ymin = tuple([ int(v) for v in box[:2] ])
    xmax, ymax = xmin+mask.shape[1], ymin+mask.shape[0]
    canvas[ymin:ymax, xmin:xmax, :] = mask
    frame[:, :, :] = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

def draw_box(frame, box, color=(0, 0, 255), thickness=2):
    """Draw bounding box on the frame

    Args:
        frame (ndarray): target frame to draw box on
        box (list): bounding box in tlbr format in pixel coordinate
        color (tuple): color of border of bounding box
        thickness (int): thickness of border of bonuding box
    """
    xmin, ymin = tuple([ int(v) for v in box[:2] ])
    xmax, ymax = tuple([ int(v) for v in box[2:4] ])
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)

def draw_text(frame, text, position,
            fgcolor=(255, 255, 255),
            bgcolor=(0, 0, 0),
            fontScale=1, thickness=2, margin=3):
    """Draw text on the specified frame
    Args:
        frame (ndarray): processing frame
        text (string): text to render
        position (tuple): text position (xmin, ymin) in pixel coordinate
        fgcolor (tuple): BGR color palette for font color
        bgcolor (tuple): BGR color palette for background color
        fontScale (int): font scale
        thickness (int): line thickness
        margin (int): space between texts
    """
    # opencv doesn't handle `\n` in the text
    # therefore we handle it line by line
    lines = text.split('\n')
    text_widths = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][0]
                    for line in lines ]
    text_heights = [ margin*2+cv2.getTextSize(text=line,
                                    thickness=thickness,
                                    fontScale=fontScale,
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)[0][1]
                    for line in lines ]
    max_width = int(max(text_widths))
    max_height = int(max(text_heights))
    xmin = int(position[0])
    ymin = int(position[1])

    # draw background
    cv2.rectangle(frame,
            (xmin, ymin),
            (xmin+max_width, ymin+max_height*len(lines)),
            bgcolor, -1)

    # draw text line by line
    for j, line in enumerate(lines):
        cv2.putText(frame, line,
                (xmin+margin, ymin+(max_height*(j+1))-margin),
                color=fgcolor,
                fontScale=fontScale,
                thickness=thickness,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

