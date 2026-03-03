# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 20:18:16 2025

@author: shrim
"""

# Loading Dicom files Shoulder Data

import numpy as np
import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as plt
from skimage.transform import resize
import glob
import os
import cv2
from PIL import Image

# Define the location of the directory
image_path =r"POCUS Shoulder Data/Final US Data/1/1_Philips/001_left.dcm"

ds = dicom.dcmread(image_path)
image = ds.pixel_array

'''for i in range(image.shape[0]):
    plt.imshow(image[i])     
    print(i)
    plt.title(f"Frame {i}")
    plt.axis("off")
    plt.show()'''

gray_frames = []

for i in range(image.shape[0]):
    frame = image[i]                       
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frames.append(gray)

gray_frames = np.stack(gray_frames)

for i in range(gray_frames.shape[0]):
    plt.imshow(gray_frames[i], cmap='gray')     
    print(i)
    plt.title(f"Frame {i}")
    plt.axis("off")
    plt.show()




