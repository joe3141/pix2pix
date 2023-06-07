import glob
import os
import shutil
from tqdm import tqdm


data_root = "/home/elab/Documents/data_Youssef/10x_ds_scans_orig_AIMOS/"
destination = "/home/elab/Documents/data_Youssef/paired_pix2pix/"

mice = sorted(os.listdir(data_root))

train_mice = mice[:12]
test_mice = mice[12:]

val_mice = train_mice[11:]
train_mice = train_mice[:11]


def copy_set(folder_names, set_label):
    pics_A = sorted(glob.glob(os.path.join(data_root, train_mice[0], "C00", "*")),
                    key=lambda x: x.split("/")[-1].split(".")[0])  # C00 C01 label
    pics_B = sorted(glob.glob(os.path.join(data_root, train_mice[0], "C01", "*")),
                    key=lambda x: x.split("/")[-1].split(".")[0])

    print("Copying " + set_label)
    for pic_A, pic_B in tqdm(zip(pics_A, pics_B)):
        shutil.copy(pic_A, os.path.join(destination, "A", set_label))
        shutil.copy(pic_B, os.path.join(destination, "B", set_label))


copy_set(train_mice, "train")
copy_set(val_mice, "val")
copy_set(test_mice, "test")
