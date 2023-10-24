"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from models import create_model
from util import util
import glob
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2

mouse_max  = 65535.0


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def __make_power_2(img, base):
    method = Image.BICUBIC
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def detorchify(img):
    img = (img + 1.0) / 2.0
    img = img.transpose((1, 2, 0))

    return img


def prepare_img(img):
    # img = img.convert('RGB')  # will be removed

    transform_list = []

    # expected behavior is that it should return the image as it is?
    # This was previously checked in torch's source code.
    # transform_list.append(transforms.Grayscale(1))  # might be removed
    # transform_list.append(transforms.Resize((768, 1280), Image.BICUBIC))
    # transform_list.append(transforms.Lambda(lambda _img: __make_power_2(_img, base=4)))  # will be removed
    transform_list.append(transforms.Lambda(lambda x: np.array(x).astype(np.float32)))
    transform_list.append(transforms.Lambda(lambda x: x / mouse_max))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5,), (0.5,))]

    _transforms = transforms.Compose(transform_list)

    return _transforms(img).unsqueeze(0)


def dir2tensors(dir):
    # Aimos 10x 16 mice sorting
    # files = sorted(glob.glob(os.path.join(data_root, "*")),
    #                key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[1][1:]))

    # lysm sorting
    # files = sorted(glob.glob(os.path.join(data_root, "*")),
    #                key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-2][1:]))

    # #.tif sorting
    # files = sorted(glob.glob(os.path.join(dir, "*")),
    #                key=lambda x: int(x.split("/")[-1].split(".")[0]))

    # lnp sorting
    files = sorted(glob.glob(os.path.join(dir, "*")), key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[1]))

    imgs = [imageio.imread(file) for file in files]
    orig_size = imgs[0].shape
    print(orig_size)
    imgs = [Image.fromarray(cv2.resize(img, (768, 1280), interpolation=cv2.INTER_CUBIC)) for img in imgs]
    # imgs = [Image.fromarray(img) for img in imgs]
    tensors = [prepare_img(img) for img in imgs]

    return tensors, files


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    data_root = "/home/elab/projects/data/LNP-mice/unpacked_norm"
    output_dir = "/home/elab/projects/data/LNP-mice/unpacked_PI"

    for mouse in os.listdir(data_root):

        mouse_output_dir = os.path.join(output_dir, mouse)
        if not os.path.exists(mouse_output_dir):
            os.makedirs(mouse_output_dir)

        tensors, files = dir2tensors(os.path.join(data_root, mouse))

        for path, data in tqdm(zip(files, tensors)):
            input_object = {"A": data, "B": data, "A_paths": path, "B_paths": path}
            model.set_input(input_object)  # unpack data from data loader
            model.test()           # run inference

    #        invTrans = transforms.Compose([transforms.Normalize(mean=[0.,], std=[1 / 0.5,]),
    #                                       transforms.Normalize(mean=[-0.5,], std=[1.,]),
    #                                       ])
            output = model.fake_B
            # output = invTrans(output)
            output_img = util.tensor2im(output, no_process=True)
            # output_img = cv2.resize(output_img, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_CUBIC)
            # print(np.unique(output_img))
            # output_img = output_img.astype(np.uint16)
            imageio.imsave(os.path.join(mouse_output_dir, path.split("/")[-1]), np.squeeze(output_img))