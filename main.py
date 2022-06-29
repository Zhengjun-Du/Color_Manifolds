# -*- coding: utf-8 -*-
## @package som_cm.results.som_single_image
#
#  Demo for single image.
#  @author      tody
#  @date        2015/08/31

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from image import loadRGB
from hist_3d import Hist3D
from som import SOMParam, SOM, SOMPlot
#from window import showMaximize


## Setup SOM in 1D and 2D for the target image.
def setupSOM(image, random_seed=100, num_samples=1000):
    np.random.seed(random_seed)

    hist3D = Hist3D(image, num_bins=16)
    color_samples = hist3D.colorCoordinates() #color_samples：1000个采样bin的颜色

    random_ids = np.random.randint(len(color_samples) - 1, size=num_samples)
    samples = color_samples[random_ids] #打乱顺序

    param1D = SOMParam(h=64, dimension=1)
    som1D = SOM(samples, param1D)

    param2D = SOMParam(h=32, dimension=2)
    som2D = SOM(samples, param2D)
    return som1D, som2D


## Demo for the single image file.
def singleImageResult(image_file = "apple.png"):
    print("image_file",image_file)
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]

    image = loadRGB(image_file)

    som1D, som2D = setupSOM(image)

    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.2)

    font_size = 15
    fig.suptitle("SOM-Color Manifolds for Single Image", fontsize=font_size)

    plt.subplot(231)
    h, w = image.shape[:2]
    plt.title("Original Image: %s x %s" % (w, h), fontsize=font_size)
    plt.imshow(image)
    plt.axis('off')

    print ("  - Train 1D")
    som1D.trainAll()

    print ("  - Train 2D")
    som2D.trainAll()

    som1D_plot = SOMPlot(som1D)
    som2D_plot = SOMPlot(som2D)
    plt.subplot(232)
    plt.title("SOM 1D", fontsize=font_size)
    som1D_plot.updateImage()
    plt.axis('off')

    plt.subplot(233)
    plt.title("SOM 2D", fontsize=font_size)
    som2D_plot.updateImage()
    plt.axis('off')

    ax1D = fig.add_subplot(235, projection='3d')
    plt.title("1D in 3D", fontsize=font_size)
    som1D_plot.plot3D(ax1D)

    ax2D = fig.add_subplot(236, projection='3d')
    plt.title("2D in 3D", fontsize=font_size)
    som2D_plot.plot3D(ax2D)

    _root_dir = os.path.dirname(__file__)
    result_file = os.path.join(_root_dir, "SOM_result.png")
    plt.savefig(result_file)


if __name__ == '__main__':
    image_file = "apple.png"
    singleImageResult(image_file)
