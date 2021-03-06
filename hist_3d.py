
# -*- coding: utf-8 -*-
## @package som_cm.core.hist_3d
#
#  Implementation of 3D color histograms.
#  @author      tody
#  @date        2015/08/28

import numpy as np
from image import * 

## Implementation of 3D color histograms.
class Hist3D:
    ## Constructor
    #  @param image          input image.
    #  @param num_bins       target number of histogram bins.
    #  @param alpha          low density clip.
    #  @param color_space    target color space. 'rgb' or 'Lab' or 'hsv'.
    def __init__(self, image,
                 num_bins=16, alpha=0.1, color_space='rgb'):
        self._computeTargetPixels(image, color_space)

        self._num_bins = num_bins
        self._alpha = alpha
        self._color_space = color_space

        self._computeColorRange()
        self._computeHistogram()

        self._plotter = Hist3DPlot(self)

    ## Plot histogram with the given density size range.
    def plot(self, ax, density_size_range=[10, 100]):
        self._plotter.plot(ax, density_size_range)

    def colorSpace(self):
        return self._color_space

    def colorIDs(self):
        color_ids = np.where(self._histPositive())
        return color_ids

    def colorCoordinates(self):
        color_ids = self.colorIDs() #每一列是一个非空bin的坐标
        num_bins = self._num_bins
        color_range = self._color_range
        return colorCoordinates(color_ids, num_bins, color_range)

    def colorDensities(self):
        return colorDensities(self._hist_bins)

    def rgbColors(self):
        return rgbColors(self._hist_bins, self._color_bins)

    def colorRange(self):
        return self._color_range

    def _computeTargetPixels(self, image, color_space):
        color_pixels = ColorPixels(image)
        self._pixels = color_pixels.pixels(color_space)
        self._rgb_pixels = color_pixels.rgb()

    def _computeColorRange(self):
        pixels = self._pixels
        cs = pixels.shape[1]

        c_min = np.zeros(cs)
        c_max = np.zeros(cs)
        for ci in range(cs):
            c_min[ci] = np.min(pixels[:, ci])
            c_max[ci] = np.max(pixels[:, ci])

        self._color_range = [c_min, c_max]

    def _computeHistogram(self):
        pixels = self._pixels
        num_bins = self._num_bins
        c_min, c_max = self._color_range

        hist_bins = np.zeros((num_bins, num_bins, num_bins), dtype=np.float32)
        color_bins = np.zeros((num_bins, num_bins, num_bins, 3), dtype=np.float32)

        color_ids = (num_bins - 1) * (pixels - c_min) / (c_max - c_min)
        color_ids = np.int32(color_ids)

        for pi, color_id in enumerate(color_ids):
            hist_bins[color_id[0], color_id[1], color_id[2]] += 1
            color_bins[color_id[0], color_id[1], color_id[2]] += self._rgb_pixels[pi]

        self._hist_bins = hist_bins  #记录每个bin的像素数目
        hist_positive = self._hist_bins > 0.0 #非空bin的数目

        for ci in range(3):
            color_bins[hist_positive, ci] /= self._hist_bins[hist_positive] #每个bin的平均颜色

        self._color_bins = color_bins

        self._clipLowDensity()

    def _clipLowDensity(self):
        clipLowDensity(self._hist_bins, self._color_bins, self._alpha)

    def _histPositive(self):
        x = self._hist_bins > 0.0
        return x


## 3D color histogram plotter.
class Hist3DPlot:
    ## Constructor.
    #  @param hist3D histogram for plotting.
    def __init__(self, hist3D):
        self._hist3D = hist3D

    ## Plot histogram with the given density size range.
    def plot(self, ax, density_size_range=[10, 100]):
        color_samples = self._hist3D.colorCoordinates()
        density_sizes = self._densitySizes(density_size_range)
        colors = self._hist3D.rgbColors()

        ax.scatter(color_samples[:, 0], color_samples[:, 1], color_samples[:, 2], color=colors, s=density_sizes)
        self._axisSetting(ax)

    def _densitySizes(self, density_size_range):
        color_densities = self._hist3D.colorDensities()
        return densitySizes(color_densities, density_size_range)

    def _axisSetting(self, ax):
        color_space = self._hist3D.colorSpace()

        ax.set_xlabel(color_space[0])
        ax.set_ylabel(color_space[1])
        ax.set_zlabel(color_space[2])

        color_range = self._hist3D.colorRange()
        tick_range = np.array(color_range).T

        xticks, yticks, zticks = range2ticks(tick_range)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks)

        xlim, ylim, zlim = range2lims(tick_range)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

def colorCoordinates(color_ids, num_bins, color_range):
    color_ids = np.array(color_ids).T
    c_min, c_max = color_range
    color_coordinates = c_min + (color_ids * (c_max - c_min)) / float(num_bins - 1.0)
    return color_coordinates


def colorDensities(hist_bins):
    hist_positive = hist_bins > 0.0
    color_densities = np.float32(hist_bins[hist_positive])

    density_max = np.max(color_densities)
    color_densities = color_densities / density_max

    return color_densities


def rgbColors(hist_bins, color_bins):
    hist_positive = hist_bins > 0.0

    colors = color_bins[hist_positive, :]
    colors = np.clip(colors, 0.0, 1.0)
    return colors


def clipLowDensity(hist_bins, color_bins, alpha):
    density_mean = np.mean(hist_bins)
    low_density = hist_bins < density_mean * alpha
    hist_bins[low_density] = 0.0

    for ci in range(3):
        color_bins[low_density, ci] = 0.0


def densitySizes(color_densities, density_size_range):
    density_size_min, density_size_max = density_size_range
    density_size_factor = density_size_max / density_size_min
    density_sizes = density_size_min * np.power(density_size_factor, color_densities)
    return density_sizes


def range2ticks(tick_range, decimals=1):
    ticks = np.around(tick_range, decimals=decimals)
    ticks[ticks > 10] = np.rint(ticks[ticks > 10])
    return ticks


def range2lims(tick_range):
    unit = 0.1 * (tick_range[:, 1] - tick_range[:, 0])
    lim = np.array(tick_range)
    lim[:, 0] += -unit
    lim[:, 1] += unit

    return lim