from tqdm import tqdm
import os
import cv2
import numpy as np
import nibabel as nib

def readNifti(path,reorient=None):
    '''
    volume = readNifti(path)

    Reads in the NiftiObject saved under path and returns a Numpy volume.
    This function can also read in .img files (ANALYZE format).
    '''
    if(path.find('.nii')==-1 and path.find('.img')==-1):
        path = path + '.nii'
    print(path)
    if(os.path.isfile(path)):
        NiftiObject = nib.load(path)
    elif(os.path.isfile(path + '.gz')):
        NiftiObject = nib.load(path + '.gz')
    else:
        raise Exception("No file found at: "+path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    if(reorient=='uCT_Rosenhain' and path.find('.img')):
        # Only perform this when reading in raw .img files
        # from the Rosenhain et al. (2018) dataset
        #    y = from back to belly
        #    x = from left to right
        #    z = from toe to head
        volume = np.swapaxes(volume,0,2) # swap y with z
        volume = np.flip(volume,0) # head  should by at y=0
        volume = np.flip(volume,2) # belly should by at x=0
    return volume


folder_in = "/home/elab/projects/data/LNP-mice/Downsampled_pred_714/"
folder_out = "/home/elab/projects/data/LNP-mice/unpacked_pred_714/"

for sample in tqdm(os.listdir(folder_in)):
    if '.nii' in sample:
        name = sample.replace('.nii', '').replace('.gz', '')
        path_out = folder_out + name
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        vol_in = readNifti(folder_in + sample)

        if ('000') in sample:
            vol_in = 65535 * vol_in

            vol_in = vol_in.astype(np.uint16)
        else:
            vol_in = vol_in.astype(np.uint8)

        for zslice in range(vol_in.shape[2]):
            cv2.imwrite(path_out + '/Zslice_' + str(zslice).zfill(4) + '.tiff', vol_in[:, :, zslice])