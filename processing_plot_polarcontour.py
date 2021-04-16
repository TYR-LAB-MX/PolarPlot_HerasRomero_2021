#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# ------------------------------
# Name:        processing_plot_plarcontour.py
# Purpose:     Processing of the maximum projected images and creation
#              of the polar plot of fluorescence intensity.
# 
# @uthor:      acph - dragopoot@gmail.com, apoot@ifc.unam.mx
#
# Created:     
# Copyright:   (c) acph 2020
# Licence:     GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007
# ------------------------------
"""Processing of the maximum projected images and creation of the polar plot
of fluorescence intensity.

The processing of the images was conducted as follows:

1. Files R1 and R2 were rotated in order to maintain the same brain orientation
   in all four files.
2. The maximum projected images were mean-binned (20 square pixels) in order to
   reduce artifacts and use fewer pixels to process. 
3. Median noise reduction was applied (disk radio = 2 pixels).
4. An intensity threshold was applied to images at percentile 90 or 95, and the
   cartesian coordinates of the pixels that surpass the threshold were obtained.
5. The origin of the coordinates was moved to the first branch split of the M4 
   segment in the superior trunk of the MCA and the coordinates where transformed
   to polar coordinates.
6. The polar coordinates where rotated as required for the approximate alignment
   of the M4 branches (R1 and R2 files) in all files.
7. The polar contour plot was created.

"""

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
from matplotlib.colors import LinearSegmentedColormap


def cart2polar(x, y):
    """Convert from cartesian to polar
    https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    r = (x**2 + y**2)**0.5
    theta = np.arctan2(y, x)   # same as np.arctan(y/x)
    return r, theta


def binimage(image, binsize, function='mean'):
    """Binns an image and apply a function by bin.
    Drops the remaining pixels if image.shape/binsize != 0
 
    Parameters
    ----------
    image : 2D np.array
        Image

    binsize : int
        Size of the bin

    function : str  mean|median|max
        Function for bin agregation

    Returns
    -------
    out : 2D np.array
        Binned image

    """
    ny, nx = image.shape
    nbinsy = ny//binsize
    nbinsx = nx//binsize
    yremain = ny - nbinsy*binsize
    xremain = nx - nbinsx*binsize
    image2bin = image[yremain:, xremain:]
    # print(ny, nx)
    # print(yremain, xremain)
    # print(image2bin.shape)
    binned = image2bin.reshape((nbinsy, binsize, nbinsx, binsize))
    if function == 'mean':
        binned = np.mean(binned, (-1, 1))
    elif function == 'median':
        binned = np.median(binned, (-1, 1))
    elif function == 'max':
        binned = np.max(binned, (-1, 1))
    else:
        raise
    return binned


# infiles maxpoj image in numpy array
images = glob("*maxproj*npy")
images.sort()

# Filtering parameter
disk_size = 2
bin_size = 20
bin_method = 'median'
maxval = 2**12
percentile = 95

figure_centers = {'R1': (134, 131),  # (135, 125),
                  'R2': (196, 147),  # (190, 120),
                  'R3': (188, 127),  # (185, 135),
                  'R4': (167,134)   # (165, 140)
                  }

colors = {'R1': [i/255 for i in (1, 35, 75)],
          'R2': [i/255 for i in (3, 57, 108)],
          'R3': [i/255 for i in (0, 91, 150)],
          'R4': [i/255 for i in (100, 151, 177)]
          }

names = {'R1': "Intact",
         'R2': "Control",
         'R3': "NxEV",
         'R4': "HxEv"
}

cmaps = {}                      # Color maps for contour plot
for key, val in colors.items():
    cmap_ = LinearSegmentedColormap.from_list(key, [val+[1],
                                                    val+[0.8],
                                                    val+[0.6]])
    cmaps[key] = cmap_

# set polar plot
fig = plt.figure(figsize=(8, 8))
pos = 220

for imagef in images:
    print(f'[INFO] Working with file {imagef} ...')
    image = np.load(imagef)
    image = image[:, :, 0]  # removing last dimention

    # inverting images R1, R2
    if 'R1-1' in imagef or 'R2-2' in imagef:
        image = image[::-1, ::-1]

    # tag to recognize the file
    tag = imagef.split('-')[0]

    # Binnig image
    print(f'    ...binning image (bin size = {bin_size})...')
    bined = binimage(image, bin_size, function=bin_method)

    # Median reduction
    print(f'     ...median noise reduction (disk size = {disk_size})...')
    reduction = median(bined, disk(disk_size))
    img = reduction

    # saving proccesed image
    print(f'     ...saving processed file.')
    np.save(f'{tag}_rotated_binning_reduced', reduction)

    # geting thresholds
    print('    ...Thresholding and processing.')
    if tag == 'R1':
        threshold = np.percentile(img, percentile-5)   # using percentile 90
    else:
        threshold = np.percentile(img, percentile)
    # thresholding
    intensity = img[img >= threshold]
    norm_int = intensity/maxval * 10

    # get coordinnates
    print('   ...Transforming coordinates.')
    y, x = np.where(img >= threshold)

    # translate coordinates to "center"
    x_, y_ = figure_centers[tag]
    x = x - x_
    y = y - y_

    # transform coordinates
    r, theta = cart2polar(x, y)
    # Rotating images to match. Only R1 and R2
    # values where obtained by hand
    if tag == 'R2':
        theta += 0.06981        # radians
    elif tag == 'R1':
        theta += 0.3839

    # round theta values to 1 decimal
    roundthetha = theta.round(1)

    # create a range of theta values to meshgrid
    thetas_r = np.arange(roundthetha.min(), roundthetha.max() + 0.1, 0.1)

    # round radii to tens
    roundr = r.round(-1)

    # create a rane of radii values to meshgrid
    r_r = np.arange(roundr.min(), 200+10, 10)

    # mesh gird
    Ts_r, Rs_r = np.meshgrid(thetas_r, r_r)

    # new matrix for 0, 1
    Z = np.zeros(Ts_r.shape) - 1

    # add 1 to the positions in Z
    for i in range(len(roundthetha)):
        ti = roundthetha[i]
        ri = roundr[i]
        tmask = thetas_r.round(1) == ti
        rmask = r_r.round(-1) == ri
        Z[rmask, tmask] = 1


    pos += 1
    ax = fig.add_subplot(pos, polar=True)
    # contour plot
    print("    ...Creating polar plot.")
    ax.contourf(Ts_r, Rs_r, Z, levels=[0.1, 0.5, 1],
                cmap=cmaps[tag], extend='neither')

    
    ax.set_rlim((0, 220))
    ax.set_rticks(range(0, 220, 50))
    ax.set_title(names[tag], loc='left')
    # plot
    #ax.scatter(theta, r, s=1, alpha=0.1, label=tag)

fig.tight_layout()
print("[INFO] Saving plot (png and svg)")
plt.savefig(f'polar_t_{percentile}_contour.png', dpi=600)
plt.savefig(f'polar_t_{percentile}_contour.svg', dpi=600)

print('[DONE] :D')
