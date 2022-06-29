# -*- coding: utf-8 -*-
## @package som_cm.cv.image
#
#  OpenCV image functions.
#  @author      tody
#  @date        2015/07/30


import numpy as np
import cv2


## Convert image into uint8 type.
def to8U(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(np.uint8(255.0 * img), 0, 255)


## Convert image into float32 type.
def to32F(img):
    if img.dtype == np.float32:
        return img
    return (1.0 / 255.0) * np.float32(img)


## RGB channels of the image.
def rgb(img):

    if len(img.shape) == 2:
        h, w = img.shape
        img_rgb = np.zeros((h, w, 3), dtype=img.dtype)
        for ci in range(3):
            img_rgb[:, :, ci] = img
        return img_rgb

    h, w, cs = img.shape
    if cs == 3:
        return img

    img_rgb = np.zeros((h, w, 3), dtype=img.dtype)

    cs = min(3, cs)

    for ci in range(cs):
        img_rgb[:, :, ci] = img[:, :, ci]
    return img_rgb


## Alpha channel of the image.
def alpha(img):
    if len(img.shape) == 2:
        return None

    cs = img.shape[2]
    if cs != 4:
        return None
    return img[:, :, 3]


## Set alpha for the image.
def setAlpha(img, a):
    h = img.shape[0]
    w = img.shape[1]

    img_rgb = None
    if len(img.shape) == 2:
        img_rgb = gray2rgb(img)
    else:
        img_rgb = img

    img_rgba = np.zeros((h, w, 4), dtype=img.dtype)
    img_rgba[:, :, :3] = img_rgb
    img_rgba[:, :, 3] = a
    return img_rgba


## RGB to Gray.
def rgb2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


## Gray to RGB.
def gray2rgb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return gray


## Gray to RGBA.
def gray2rgba(img):
    gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    return gray


## BGR to RGB.
def bgr2rgb(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


## BGRA to RGBA.
def bgra2rgba(img):
    a = alpha(img)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    if a is not None:
        rgba[:, :, 3] = a
    return rgba


## RGBA to BGRA.
def rgba2bgra(img):
    a = alpha(img)
    bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    bgra[:, :, 3] = a
    return bgra


## RGB to BGR.
def rgb2bgr(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


## RGB to Lab.
def rgb2Lab(img):
    img_rgb = rgb(img)
    Lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return Lab


## Lab to RGB.
def Lab2rgb(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return rgb


def rgb2hsv(img):
    img_rgb = rgb(img)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


## HSV to RGB.
def hsv2rgb(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return rgb


## Merge channels.
def merge(channels):
    cs = 0
    h = 0
    w = 0
    for channel in channels:
        if len(channel.shape) == 2:
            cs += 1
        else:
            cs += channel.shape[2]

        h, w = channel.shape[0], channel.shape[1]

    img = np.zeros((h, w, cs))

    ci = 0
    for channel in channels:
        if len(channel.shape) == 2:
            img[:, :, ci] = channel[:, :]
            ci += 1
            continue

        for cci in range(channel.shape[2]):
            img[:, :, ci] = channel[:, :, cci]
            ci += 1

    return img


## Luminance value from Lab.
#  Lumiannce value will be in [0, 1]
def luminance(img):
    L = rgb2Lab(rgb(img))[:, :, 0]
    if L.dtype != np.uint8:
        return (1.0 / 100.0) * L
    return L


def loadGray(file_path):
    bgr = cv2.imread(file_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def loadRGB(file_path):
    bgr = cv2.imread(file_path)
    if bgr is None:
        return None
    return bgr2rgb(bgr)


def loadRGBA(file_path):
    bgra = cv2.imread(file_path, -1)
    if bgra is None:
        return None
    return bgra2rgba(bgra)


def loadAlpha(file_path):
    bgra = cv2.imread(file_path, -1)
    return alpha(bgra)


def saveRGBA(file_path, img):
    bgra = rgba2bgra(img)
    cv2.imwrite(file_path, bgra)


def saveRGB(file_path, img):
    bgr = rgb2bgr(img)
    cv2.imwrite(file_path, bgr)


def saveGray(file_path, img):
    rgbImg = rgb(img)
    cv2.imwrite(file_path, rgbImg)


def saveImage(file_path, img):
    img_8U = to8U(img)

    if len(img_8U.shape) == 2:
        saveGray(file_path, img_8U)
        return

    if img_8U.shape[2] == 3:
        saveRGB(file_path, img_8U)
        return

    if img_8U.shape[2] == 4:
        saveRGBA(file_path, img_8U)
        return



## True if x is a vector.
def isVector(x):
    return x.size == x.shape[0]


## True if x is a matrix.
def isMatrix(x):
    return not isVector(x)


## Normalize vector.
def normalizeVector(x):
    norm = np.linalg.norm(x)
    y = x
    if norm > 0:
        y = np.ravel((1.0 / norm) * x)
    return y


## Normalize vectors (n x m matrix).
def normalizeVectors(x):
    norm = normVectors(x)
    nonZeroIDs = norm > 0
    x[nonZeroIDs] = (x[nonZeroIDs].T / norm[nonZeroIDs]).T
    return x


## Norm of vectors (n x m matrix).
def normVectors(x):
    return np.sqrt(l2NormVectors(x))


## L2 norm of vectors (n x m matrix).
#  n x 1 vector: call np.square.
#  n x m vectors: call np.einsum.
def l2NormVectors(x):
    if isVector(x):
        return np.square(x)
    else:
        return np.einsum('...i,...i', x, x)



class ColorPixels:
    ## Constructor
    #  @param image          input image.
    #  @param num_pixels     target number of pixels from the image.
    def __init__(self, image, num_pixels=1000):
        self._image = to32F(image)
        self._num_pixels = num_pixels
        self._rgb_pixels = None
        self._Lab = None
        self._hsv = None

    ## RGB pixels.
    def rgb(self):
        if self._rgb_pixels is None:
            self._rgb_pixels = self.pixels("rgb")
        return self._rgb_pixels

    ## Lab pixels.
    def Lab(self):
        if self._Lab is None:
            self._Lab = self.pixels("Lab")
        return self._Lab

    ## HSV pixels.
    def hsv(self):
        if self._hsv is None:
            self._hsv = self.pixels("hsv")
        return self._hsv

    ## Pixels of the given color space.
    def pixels(self, color_space="rgb"):
        image = np.array(self._image)
        if color_space == "rgb":
            if _isGray(image):
                image = gray2rgb(image)

        if color_space == "Lab":
            image = rgb2Lab(self._image)

        if color_space == "hsv":
            image = rgb2hsv(self._image)
        return self._image2pixels(image)

    def _image2pixels(self, image):
        if _isGray(image):
            h, w = image.shape
            step = h * w / self._num_pixels
            return image.reshape((h * w))[::step]

        h, w, cs = image.shape
        step = (int)(h * w / self._num_pixels)
        return image.reshape((-1, cs))[::step]


def _isGray(image):
    return len(image.shape) == 2
