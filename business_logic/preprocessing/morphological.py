import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MinMaxScaler

def show(im):
    plt.imshow(im, cmap='gray')
    plt.show()

def invert(im):
    return (im*-1)+255

def min_max_scale(im):
    im = MinMaxScaler().fit_transform(im.ravel().reshape(-1,1))
    return im.reshape(300,300)

def brighten(im):
    m = np.max(im)-.3
    im[im >= m] += 0.3
    im[im < m] -= 0.3
    return im

def blur(im, kernel):
    im = im.astype("uint8")
    im = cv2.medianBlur(im, kernel)
    return im

def morph_close(im, kernel, i):
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=i)

def morph_open(im, kernel, i):
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=i)

def morph_dilate(im, kernel, i):
    return cv2.dilate(im, kernel, iterations=i)

def morph_erode(im, kernel, i):
    return cv2.erode(im, kernel, iterations=i)

def transform(im):
    dest = np.float32([[0,0],[300,0],[0,300],[300,300]])
    source = np.float32([[25,25],[275,25],[25,275],[275,275]])
    res = cv2.getPerspectiveTransform(source,dest)
    return cv2.warpPerspective(im, res, (300,300))

def vert_det(im, kernel):
    vert = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=kernel)
    return im - vert

def hor_det(im, kernel):
    hor = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=kernel)
    return im - hor

def binarize(im):
    threshold = np.max(im) - .6
    im[im < threshold] = 0
    im[im >= threshold] = 1
    return im

def otsu(im):
    im = im.astype("uint8")
    _, r = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return r

def bounding_box_transform(im):
    tt = np.argmax(im, axis=0)
    leftbound = np.amin(np.nonzero(tt))
    rightbound = np.amax(np.nonzero(tt))
    t = np.argmax(im, axis=1)
    upbound = np.amin(np.nonzero(t))
    downbound = np.amax(np.nonzero(t))
    boundarybox = np.float32([[leftbound - 10, upbound - 10], [rightbound + 10, upbound - 10]
                                 , [leftbound - 10, downbound + 10], [rightbound + 10, downbound + 10]])
    newbox = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(boundarybox, newbox)
    dst = cv2.warpPerspective(im, M, (300, 300))
    return dst

# takes in a non-inverted image (non min max normalized)
def remove_lines(z):
    z = z.reshape(300, 300)

    # threshold the image
    thresh = cv2.threshold(z, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(z, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - z, cv2.MORPH_CLOSE, repair_kernel, iterations=2)

    # threshold the image
    thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove vertical
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
    result = 255 - cv2.morphologyEx(255 - result, cv2.MORPH_CLOSE, repair_kernel, iterations=2)

    return result

