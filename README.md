# Polar plots

Heras-Romero, et.al.

This repository contains the scripts used for the processing of whole-brain confocal tiles images to create a polar plot of fluorescent intensity.

The processing of confocal images was separated into two scripts, as explained below. 

## 1. get_max_projection.py
This script gets the maximum projection from the raw Zeiss confocal .czi files. 
Due to the large size of the original files, this script must be executed in a
computer with more than 16 GB of RAM.

### Dependencies.
	- Python 3
	- czifile 
	- numpy

### Input.
	- Original .czi files form Zeiss confocal microscope (four images, one for
	  each experimental condition).
	
### Output.
	- Maximum projected images in numpy array binary files (*.npy).

## 2. processing_plot_polarcontour.py
This script process the maximum projected images and creates the final polar plot
of fluorescence intensity.

### Dependencies.
	- numpy
	- matplotlib
	- scikit-image

### Input.
	- Maximum projected images created by the previous script (*.npy files)

### Output.
	- Polar plot of fluorescence intensity (.png and .svg).

### Image processing.

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

