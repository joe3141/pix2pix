import glob
import os


data_root = "/home/elab/Documents/data_Youssef/10x_ds_scans_orig_AIMOS/"

mice = sorted(os.listdir(data_root))

train_mice = mice[:12]
test_mice = mice[12:]

val_mice = train_mice[11:]
train_mice = train_mice[:11]

pics = sorted(glob.glob(os.path.join(data_root, train_mice[0], "C00", "*"))) # C00 C01 label