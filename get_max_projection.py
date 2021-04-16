#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# ------------------------------
# Name:        get_max_projection.py
# Purpose:     Save maximum projection of *.czi stack files in the current folder.
# 
# @uthor:      acph - dragopoot@gmail.com, apoot@ifc.unam.mx
#
# Created:     
# Copyright:   (c) acph 2020
# Licence:     GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007
# ------------------------------
"""Save max projection of *.czi stack files in the current folder.

WARNING: As the the image stacks may be huge, it is recommended to do this step
in a computer with more than 16 GB of RAM.
"""

from glob import glob
import numpy as np
import czifile

# get czi files in current folder
files = glob('*.czi')

for imfile in files:
    print(f"[INFO] reading file: {imfile}")
    image = czifile.imread(imfile)[0, 0, 0, 0]
    print(f"      ...   {image.shape}")

    # project and delet original image
    proj = image.max(0)
    del image

    # save max_proj
    np.save(f'{imfile}_maxproj.np', proj)
    print(f'[INFO] max projection saved')

print("[DONE] :D")

