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


def process_file_set(channel_files, label):
    global n
    a, b, c = channel_files
    print(f"Processing {label} set")
    for a_file, b_file, c_file in tqdm(zip(a, b, c)):
        img_a = imageio.imread(a_file)
        img_b = imageio.imread(b_file)
        img_c = imageio.imread(c_file)
        for piece_a, piece_b, piece_c in zip(sliding_window_on_image(img_a, width, height, overlap=0),
                                             sliding_window_on_image(img_b, width, height, overlap=0),
                                             sliding_window_on_image(img_c, width, height, overlap=0)):
            image_crop_a, i_index, j_index = piece_a
            image_crop_b, i_index, j_index = piece_b
            image_crop_c, i_index, j_index = piece_c
            image_crop_a = np.array(image_crop_a)
            if IsImageFG(image_crop_a):
                imageio.imsave(os.path.join(data_root, "A_c", label, f"{n}.tif"), image_crop_a)
                imageio.imsave(os.path.join(data_root, "B_c", label, f"{n}.tif"), image_crop_b)
                imageio.imsave(os.path.join(data_root, "C_c", label, f"{n}.tif"), image_crop_c)
                n += 1
                

def get_channel_files(channel_label):
    train_files = glob.glob(os.path.join(data_root, channel_label, "train", "*"))
    val_files = glob.glob(os.path.join(data_root, channel_label, "val", "*"))
    test_files = glob.glob(os.path.join(data_root, channel_label, "test", "*"))

    return train_files, val_files, test_files


A_sets = get_channel_files("A")
B_sets = get_channel_files("B")
C_sets = get_channel_files("C")

process_file_set((A_sets[0], B_sets[0], C_sets[0]), "train")
process_file_set((A_sets[1], B_sets[1], C_sets[1]), "val")
process_file_set((A_sets[2], B_sets[2], C_sets[2]), "test")


