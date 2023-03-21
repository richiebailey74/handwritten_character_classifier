import cv2
from .morphological import *

def preprocess(ims):
    r = []
    for im in ims.T:
        im = im.reshape((300,300))
        im = transform(im)
        im = invert(im)
        im = blur(im, 5)
        im = morph_erode(im, (3,3), 5)
        im = morph_dilate(im, (3,3), 5)
        im = morph_close(im, (3,3), 5)
        im = morph_open(im, (3,3), 5)
        im = min_max_scale(im)
        im = brighten(im)
        im = min_max_scale(im)
        im = cv2.resize(im, (100,100))
        im = np.stack((im,)*3, axis=-1)
        r.append(im)
    return np.array(r)

def augment(ims, labels):
    result = []
    n = []
    o = []
    t = []
    for im in ims:
        im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
        n.append(im)
        im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
        o.append(im)
        im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
        t.append(im)
    result.append(ims)
    result.append(n)
    result.append(o)
    result.append(t)
    labels_new = np.concatenate((labels,labels,labels,labels))
    return np.array(result).reshape(-1,100,100), labels_new