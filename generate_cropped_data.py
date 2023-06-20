import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import imageio


data_root = "/home/elab/Documents/data_Youssef/paired_pix2pix/"

n = 0
width = 256
height = 256


def sliding_window_on_image(image_in, height, width, overlap=0):
    '''given an input image, cropping size, and overlap, return the subset of
     sub-crops from the original image, as well as the position indicators (i and j)
     of where these came from '''
    image_in = Image.fromarray(
        image_in.astype(np.uint16))  # I use this with original tiffs that are uint16, please adapt as needed
    imgwidth, imgheight = image_in.size

    steps_i = (imgheight - height) // (height - overlap) + 1
    steps_j = (imgwidth - width) // (width - overlap) + 1

    for i in range(steps_i):
        for j in range(steps_j):
            yield [image_in.crop((j * (width - overlap), i * (height - overlap), j * (width - overlap) + width,
                                  i * (height - overlap) + height)),
                   i, j]


def IsImageFG(image_in, foreground_threshold_value=15, percentage_thr=0.2):
    '''given an image and a pixel value threshold, compare if a required percentage
    of the image is larger than this threshold. If yes, the image is can be filtered as foreground'''
    if(np.sum(image_in>foreground_threshold_value)/(image_in.shape[0]*image_in.shape[1]))<percentage_thr:
        return False
    return True


def process_file_set(files, dir_label, label):
    global n
    print(f"Processing {dir_label}'s {label} set")
    for file in tqdm(files):
        img = imageio.imread(file)
        for k, piece in enumerate(sliding_window_on_image(img, width, height, overlap=0)):
            image_crop, i_index, j_index = piece
            image_crop = np.array(image_crop)
            if IsImageFG(image_crop):
                imageio.imsave(os.path.join(data_root, f"{dir_label}_c", label, f"{n}.tif"), image_crop)
                n += 1


def process_channel_dir(dir_label):
    train_files = glob.glob(os.path.join(data_root, dir_label, "train", "*"))
    val_files = glob.glob(os.path.join(data_root, dir_label, "val", "*"))
    test_files = glob.glob(os.path.join(data_root, dir_label, "test", "*"))

    process_file_set(train_files, dir_label, "train")
    process_file_set(val_files, dir_label, "val")
    process_file_set(test_files, dir_label, "test")


process_channel_dir("A")
process_channel_dir("B")
process_channel_dir("C")

