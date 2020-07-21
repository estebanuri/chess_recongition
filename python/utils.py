import cv2
import numpy as np

def resize_w(img, new_w = 640):
    w, h = img.shape[1], img.shape[0]
    new_h = int((new_w / w) * h)
    ret = cv2.resize(img, (new_w, new_h))
    return ret

def shrink_if_large(img, max=1080):
    w, h = img.shape[1], img.shape[0]
    if w > max or h > max:
        img = resize_w(img, max)

    return img

def imshow(win_label, image):
    cv2.namedWindow(win_label, cv2.WINDOW_NORMAL)
    cv2.imshow(win_label, image)

def pix(np_array):
    return tuple(np.round(np_array).astype(int))