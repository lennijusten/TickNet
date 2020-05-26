# Cropping Function
# Author: Lennart Justen
# Last revision: 5/22/20

# Description: This function crops images from a specified directory "dir_path_old"
# and saves them to a new directory specified by "dir_path_new."
# The cropping procedure crops the images into a square along their shortest sides.
# The original files remain unchanged and the same filename is used for both the original
# and cropped images.

from PIL import Image
from glob import glob
from math import floor
import numpy as np
from os.path import join, basename

dir_path_old = "/Users/Lenni/Desktop/Box Images"
dir_path_new = "/Users/Lenni/Desktop/Box Images Cropped/"


def im_crop(dir_path_old, dir_path_new):
    files = []
    for ext in ('*.png', '*.jpg', '*.JPG', '*.PNG'):
        files.extend(glob(join(dir_path_old, ext)))

    im_list_old = []
    im_list_new = []
    im_size_old = []
    im_size_new = []
    for filename in files:
        im = Image.open(filename)

        dim = im.size
        shortest = min(dim[0:2])
        longest = max(dim[0:2])

        lv = np.array(range(0, shortest)) + floor((longest - shortest) / 2)
        if dim[0] == shortest:
            im_cropped = np.asarray(im)[lv, :, :]
        else:
            im_cropped = np.asarray(im)[:, lv, :]

        im_cropped = Image.fromarray(im_cropped)
        im_cropped.save(join(dir_path_new, basename(im.filename)))

        im_list_old.append(im.filename)
        im_list_new.append(join(dir_path_new, basename(im.filename)))
        im_size_old.append(im.size)
        im_size_new.append(im_cropped.size)

    return im_list_old, im_list_new, im_size_old, im_size_new


im_list_old, im_list_new, im_size_old, im_size_new = im_crop(dir_path_old, dir_path_new)
print("Un-cropped path: ", im_list_old,sep='\n')
print("Cropped path: ", im_list_new,sep='\n')
print("Original size: ", im_size_old)
print("Cropped size: ", im_size_new)
