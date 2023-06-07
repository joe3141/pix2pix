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

index = 0


def copy_set(folder_names, set_label):
    global index
    print("Copying " + set_label)
    for folder in tqdm(folder_names):
        pics_A = sorted(glob.glob(os.path.join(data_root, folder, "C00", "*")),
                        key=lambda x: x.split("/")[-1].split(".")[0])  # C00 C01 label
        pics_B = sorted(glob.glob(os.path.join(data_root, folder, "C01", "*")),
                        key=lambda x: x.split("/")[-1].split(".")[0])

        pics_C = sorted(pics_B = sorted(glob.glob(os.path.join(data_root, folder, "label", "*")),
                                        key=lambda x: x.split("/")[-1].split(".")[0]))

        for pic_A, pic_B, pic_C in zip(pics_A, pics_B, pics_C):
            dst_filename = f"{index}.tif"
            dst_filename_label = f"{index}.tiff"

            shutil.copy(pic_A, os.path.join(destination, "A", set_label, dst_filename))
            shutil.copy(pic_B, os.path.join(destination, "B", set_label, dst_filename))
            shutil.copy(pic_C, os.path.join(destination, "C", set_label, dst_filename_label))
            
            index += 1


copy_set(train_mice, "train")
copy_set(val_mice, "val")
copy_set(test_mice, "test")
