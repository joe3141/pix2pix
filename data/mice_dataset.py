import os

import torch
import os
from glob import glob

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


def get_paths(path):
    mice = sorted(os.listdir(path))
    a_paths, b_paths = [], []
    for mouse in mice:
        if os.path.isdir(os.path.join(path, mouse)):
            with open(os.path.join(path, mouse, "meta.txt"), "r") as f:
                lines = f.readlines()
                max_a = int(lines[0])
                max_b = int(lines[1])

            a_img_files = sorted(glob(os.path.join(path, mouse, "C00", "*")), key=lambda x: x.split("/")[-1].split(".")[0])
            b_img_files = sorted(glob(os.path.join(path, mouse, "C01", "*")), key=lambda x: x.split("/")[-1].split(".")[0])

            a_img_files = [(file, max_a) for file in a_img_files]
            b_img_files = [(file, max_b) for file in b_img_files]

            a_paths += a_img_files
            b_paths += b_img_files

    return a_paths, b_paths


class MiceDataset(BaseDataset):

    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.a_paths, self.b_paths = get_paths(self.dir_AB)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        assert(len(self.a_paths) == len(self.b_paths))
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        a_path, max_a = self.a_paths[index]
        b_path, max_b = self.b_paths[index]

        A = Image.open(a_path)
        B = Image.open(b_path)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), _max=max_a)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), _max=max_b)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': a_path, 'B_paths': b_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.a_paths)
