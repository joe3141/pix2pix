import os
import cv2
import numpy as np
from libtiff import TIFFfile
from libtiff import TIFF
import re
import pickle
import nibabel as nib
from PIL import Image
from tqdm import tqdm


#%%
def psave(path, variable):
    '''
    psave(path, variable)

    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.

    # ToDo: save several variables (e.g. take X args, store them to special DICT, and save to file)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    file = open(path, 'wb')
    pickle.dump(variable,file,protocol=4)


#%%
def pload(path):
    '''
    variable = pload(path)

    Loads a variable from a file that was specified in the path. The path must at least contain the
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if
    the file is not to located in the current working directory.

    # ToDo: load several variables (e.g. load special DICT from file and return matching entries)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    file = open(path, 'rb')
    return pickle.load(file)


#%%
def stack2volume(pathtofolders,folderlist, ignoreSizeLimit=False):
    '''
    Loads stack of images from arbitrary number of folders where each folder corresponds to one
    channel. Expects each file per folder to be grayscale.

    Returns Numpy array v of dimensionality (n_y,n_x,n_z,n_c) where

    * n_y is the height of a single image
    * n_x is the width of a single image
    * n_z is the number of images per folder
    * n_c is the number of folders

    Function will not load images if more than 8 GB of RAM are required, unless flag 'ignoreSizeLimit' is set to True
    '''

    # Check whether volume exceeds RAM
    # size = estimateVolSize(pathtofolders,folderlist)
    # if ignoreSizeLimit == False and size[0] > 8*1024:
    #     raise ValueError("Error: Loading volume requires RAM of " + str(size[0]) + "MB")

    n_c = len(folderlist)
    n_z = 0
    n_x = 0
    n_y = 0
    bitdepth = None
    v = 0

    for c in range(0, n_c):
        folder = pathtofolders + '/' + folderlist[c]
        filelist = sorted(os.listdir(folder))
        if n_z == 0:
            n_z = len(filelist)
        else:
            if n_z != len(filelist): raise ValueError("Number of files in folders not consistent")
        for z in range(0,n_z):
            path = folder + '/' + filelist[z]
            image = cv2.imread(path,2)
            if n_y == 0:
                (n_y,n_x) = image.shape
            else:
                if (n_y,n_x) != image.shape: raise ValueError("Image sizes not consistent; see " + path)
            if bitdepth == None:
                bitdepth = image.dtype
            else:
                if bitdepth != image.dtype: raise ValueError("Bitdepth not consistent; see " + path)
            if type(v) == int:
                v = np.zeros((n_y,n_x,n_z,n_c),bitdepth)
            v[:,:,z,c] = image
    return v



#%%
def readFijiTIFF(path):
    '''
    (volume, FijiMeta) = readFijiTIFF(path)
        - Reads in 4D TIFF file as saved by Fiji and returns a Numpy volume (y,x,z,c) as well as a readable version of the file's meta data

    volume (Numpy array)
        * y: image height
        * x: image width
        * z: image depth (number of slices)
        * c: number of channels

    FijiMeta (dict)
        * -- stored within TiffTag 'ImageDescription' --
        * version: Fiji software version
        * nimages: total number of images (z*c)
        * nchannels: number of channels (c)
        * nslices: number of slices (z)
        * nhyperstack: 'true' if 4D volume
        * nmode: 'composite' if Fiji was used to merge channels
        * nloop: 'false'
        * nmin: minimum value for Fiji signal scaling
        * nmax: maximum value in Fiji signal scaling
        * -- stored as standalone TiffTags --
        * ImageLength: image height (y)
        * ImageWidth: image width (x)
        * PhotometricInterpretation: how to interpret pixel values (e.g. 'BlackIsZero')
        * PlanarConfiguration: (unclear)
    '''
    if(path.find('.tif')==-1):
        path = path + '.tif'
    tiff = TIFFfile(path)
    # Read out images and save in Numpy array
    samples, sample_names = tiff.get_samples()
    outList = []
    for sample in samples:
        outList.append(np.copy(sample)[...,np.newaxis])
    volume = np.concatenate(outList,axis=-1)
    volume = np.rollaxis(volume,0,3)

    # Read out Fiji meta data
    template = r'ImageJ=(?P<version>.*)\nimages=(?P<nimages>.*)\nchannels=(?P<nchannels>.*)\nslices=(?P<nslices>.*)\nhyperstack=(?P<nhyperstack>.*)\nmode=(?P<nmode>.*)\nloop=(?P<nloop>.*)\nmin=(?P<nmin>.*)\nmax=(?P<nmax>.*)\n\x00'
    ifd = tiff.get_first_ifd(0)
    FijiMeta_IDb = ifd.get_value('ImageDescription', human=True)
    FijiMeta_ID = FijiMeta_IDb.decode()
    #FijiMeta_PIh = ifd.get_value('PhotometricInterpretation', human=True) --> would give 'BlackIsZero' instead of 1
    FijiMeta_PI = ifd.get_value('PhotometricInterpretation')
    FijiMeta_PC = ifd.get_value('PlanarConfiguration')
    FijiMeta_O = ifd.get_value('Orientation')
    FijiMeta_L = ifd.get_value('ImageLength')
    FijiMeta_W = ifd.get_value('ImageWidth')
    try:
        matches = re.match(template, FijiMeta_ID)
        FijiMeta = {'version': matches.group('version'),
                    'nimages': int(matches.group('nimages')),
                    'nchannels': int(matches.group('nchannels')),
                    'nslices': int(matches.group('nslices')),
                    'nhyperstack': matches.group('nhyperstack'),
                    'nmode': matches.group('nmode'),
                    'nloop': matches.group('nloop'),
                    'nmin': matches.group('nmin'),
                    'nmax': matches.group('nmax'),
                    'PhotometricInterpretation': FijiMeta_PI,
                    'PlanarConfiguration': FijiMeta_PC,
                    'Orientation': FijiMeta_O,
                    'ImageLength': FijiMeta_L,
                    'ImageWidth': FijiMeta_W}
        # Rearrange volume from (y,x,z*c,f=1) to (y,x,z,c)
        if FijiMeta['nimages'] == FijiMeta['nchannels']*FijiMeta['nslices'] == volume.shape[2]:
            volume_old = volume
            volume = np.zeros((volume.shape[0], volume.shape[1], FijiMeta['nslices'], FijiMeta['nchannels']), volume_old.dtype.name)
            for c in range(0,FijiMeta['nchannels']):
                volume[:,:,:,c] = volume_old[:,:,c:FijiMeta['nimages']:FijiMeta['nchannels'],0]
    except:
        FijiMeta = 'Failed to read meta data'

    tiff.close()
    return(volume,FijiMeta)

#%%
def writeFijiTIFF(path, volume, FijiMeta=None, compression=None):
    '''
    Writes volume (and meta data) to 4D TIFF file that can be read as 'composite' by Fiji. Meta data does not have to be provided - default values will be used and updated to volume size if necessary. If meta data is provided, it needs to be the full dictionary with all values.

    volume (3D or 4D Numpy array)
        * y: image height
        * x: image width
        * z: image depth (number of slices)
        * c: number of channels (c=1 for 3D array)

    Optional: FijiMeta (dict)
        * -- stored within TiffTag 'ImageDescription' --
        * version: Fiji software version
        * nimages: total number of images (z*c)
        * nchannels: number of channels (c)
        * nslices: numder of slices (z)
        * nhyperstack: 'true' if 4D volume
        * nmode: 'composite' if Fiji was used to merge channels
        * nloop: 'false'
        * nmin: minimum value for Fiji signal scaling
        * nmax: maximum value in Fiji signal scaling
        * -- stored as standalone TiffTags --
        * ImageLength: image height (y)
        * ImageWidth: image width (x)
        * PhotometricInterpretation: how to interpret pixel values (e.g. 'BlackIsZero')
        * PlanarConfiguration: (unclear)
    '''

    # Convert 3D volume to 4D volume, if necessary
    if(volume.ndim == 3): volume = np.expand_dims(volume, axis=3)
    if(volume.ndim != 4): raise ValueError("Volume must be 3D or 4D")

    # Check/update meta information
    if FijiMeta == None:
        # Use default values
        status = 'No meta information provided; we therefore used default values'
        FijiMeta = {'version': '1.51u',
                    'nimages': volume.shape[2]*volume.shape[3],
                    'nchannels': volume.shape[3],
                    'nslices': volume.shape[2],
                    'nhyperstack': 'true',
                    'nmode': 'composite',
                    'nloop': 'false',
                    'nmin': np.min(volume),
                    'nmax': np.max(volume),
                    'PhotometricInterpretation': 1, # BlackIsZero
                    'PlanarConfiguration': 1, # Chunky
                    'Orientation': 1, # TopLeft
                    'ImageLength': volume.shape[0],
                    'ImageWidth': volume.shape[1]}
    else:
        delta = {'nimages': volume.shape[2]*volume.shape[3],
                 'nchannels': volume.shape[3],
                 'nslices': volume.shape[2],
                 'ImageLength': volume.shape[0],
                 'ImageWidth': volume.shape[1]}
        new = FijiMeta
        if new.update(delta) != FijiMeta: status = 'Had to update meta information to match volume'
        FijiMeta = new

    # Encode meta information
    template = 'ImageJ={version}\nimages={nimages}\nchannels={nchannels}\nslices={nslices}\nhyperstack={nhyperstack}\nmode={nmode}\nloop={nloop}\nmin={nmin}\nmax={nmax}\n\x00'
    FijiMeta_ID = template.format(**FijiMeta)
    FijiMeta_IDb = FijiMeta_ID.encode()

    # Rearrange volume from (y,x,z,c) to (y,x,z*c,f=1)
    bitdepth = int(''.join(filter(str.isdigit, str(volume.dtype.name))))
    size = volume.shape[0]*volume.shape[1]*FijiMeta['nslices']*FijiMeta['nchannels']*bitdepth / 1024**3
    if size > 4: raise ValueError("Cannot write TIFF files larger than 4GB")
    if FijiMeta['nslices'] == volume.shape[2] and FijiMeta['nchannels'] == volume.shape[3] and FijiMeta['nimages'] == volume.shape[2]*volume.shape[3]:
        volume_old = volume
        volume = np.zeros((volume.shape[0],volume.shape[1],FijiMeta['nslices']*FijiMeta['nchannels'],1),volume_old.dtype.name)
        for c in range(0,FijiMeta['nchannels']):
            volume[:,:,c:FijiMeta['nimages']:FijiMeta['nchannels'],0] = volume_old[:,:,:,c]

    # Write data to file
    if(path.find('.tif')==-1):
        path = path + '.tif'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    outTiff = TIFF.open(path, mode='w')
    outTiff.SetField('ImageDescription', FijiMeta_IDb)
    outTiff.SetField('Photometric', FijiMeta['PhotometricInterpretation']) # libtiff_ctypes.py has different nomenclature
    outTiff.SetField('PlanarConfig', FijiMeta['PlanarConfiguration']) # libtiff_ctypes.py has different nomenclature
    outTiff.SetField('Orientation', FijiMeta['Orientation'])
    outTiff.SetField('ImageLength', FijiMeta['ImageLength'])
    outTiff.SetField('ImageWidth', FijiMeta['ImageWidth'])
    volume = np.rollaxis(volume, 3, 0)
    for zInd in range(volume.shape[3]):
        outTiff.write_image(volume[:,:,:,zInd],compression=compression, write_rgb=False)

    outTiff.close()
    return None


#%%
def writeNifti(path,volume,compress=False):
    '''
    writeNifti(path,volume)

    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path.
    '''
    if(path.find('.nii')==-1 and compress==False):
        path = path + '.nii'
    if(path.find('.nii.gz')==-1 and compress==True):
        path = path + '.nii.gz'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,path)


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


#%%

def listfolders(path, searchstring=''):
    if(path[-1] != '/'):
        path = path + '/'
    folders = []
    for element in os.listdir(path):
        cond1 =  os.path.isdir(path + element) is True
        cond2 =  searchstring in element
        if(cond1 and cond2):
            folders.append(element)
    folders = sorted(folders)
    return folders


def listfiles(path, searchstring=''):
    if(path[-1] != '/'):
        path = path + '/'
    folders = []
    for element in os.listdir(path):
        cond1 =  os.path.isdir(path + element) is False
        cond2 =  searchstring in element
        if(cond1 and cond2):
            folders.append(element)
    folders = sorted(folders)
    return folders

### cropping TIFF Files

def crop_2ch_TIFF_with_overlap(image,width,height,overlap):
    '''
    crop_2ch_TIFF_with_overlap(2_channels_image, crop_width, crop_height, overlap_of_patches)

    Takes a 2 channel uint16 image, and crops it into pieces of the desired width and height, with the specified overlap
    The final shape will most likely be smaller than the input size (since it is likely not going to be an exact crop,
    unless previously calculated to be so. The boundaries for cropping are going to be orig_width and orig_height that
    serve as input for the reassembling function underneath
    returns lists of [image_crop_channel_0, image_crop_channel_1, i_index, j_index]
    (i and j index are necessary for re-assembling the images)
    '''
    image_1 = Image.fromarray(image.astype(np.uint16)[:,:,0])
    image_2 = Image.fromarray(image.astype(np.uint16)[:,:,1])
    imgwidth, imgheight = image_1.size

    steps_i = (imgheight - height) // (height - overlap) + 1
    steps_j = (imgwidth - width) // (width - overlap) + 1
    print('The boundaries for cropping are:',steps_i*(height - overlap), steps_j*(height - overlap) )
    #print('steps:', steps_i)
    for i in range(steps_i):
        for j in range(steps_j):
            yield [image_1.crop((j*(width-overlap),i*(height-overlap),j*(width-overlap)+width,i*(height-overlap)+height)),
                image_2.crop((j*(width-overlap),i*(height-overlap),j*(width-overlap)+width,i*(height-overlap)+height)),
                   i,j]

def reassemble_from_pieces_TIFFS_2ch_with_overlap(file_list,orig_width,orig_height,width,height,overlap):
    '''
    reassemble_from_pieces_TIFFS_2ch_with_overlap(list_of_list_of_images, crop_width, crop_height, overlap_of_patches)
    list_of_list_of_images = [[img_0_ch_0,img_0_ch_1,i_index_0,j_index_0],[img_1_ch_0,img_1_ch_0,i_index_1,j_index_1],... etc]

    Takes a list of 2 channel uint16 images, along with their i and j locations in the big reassembled image,
     and pastes them into place. Given that there is likely overlap, a map saves the number of times each voxel was
     passed on. For recovering the original images, the big_img outputs need to be divided by map_passed_on.
    '''
    big_img_1 = np.zeros((orig_height,orig_width), dtype = np.uint64)
    big_img_2 = np.zeros((orig_height,orig_width),dtype=np.uint64)
    map_passed_on = np.zeros((orig_height,orig_width))
    all_is = []
    all_js= []
    for file in file_list:
        image_1 = np.asarray(file[0]).astype(np.uint64)
        image_2 = np.asarray(file[1]).astype(np.uint64)
        i = file[2]
        j = file[3]

        index_i_1 = i*(height-overlap)
        index_i_2 = i*(height-overlap)+height
        index_j_1 = j*(width-overlap)
        index_j_2 = j*(width-overlap)+width
        big_img_1[index_i_1:index_i_2,index_j_1:index_j_2]+=image_1
        big_img_2[index_i_1:index_i_2,index_j_1:index_j_2]+=image_2
        map_passed_on[index_i_1:index_i_2,index_j_1:index_j_2]+=np.ones((height,width))

        all_is.append(i)
        all_js.append(j)

    return big_img_1,big_img_2,map_passed_on


def stack_of_tiffs_to_nifti(folder_in, path_out, name_out='volume.nii.gz'):
    '''given a folder of images, stack them into a 3D volume'''

    images_in = sorted(os.listdir(folder_in))
    image_for_size = cv2.imread(folder_in + images_in[0], 2)

    # volume = np.zeros((image_for_size.shape[0], image_for_size.shape[1], len(images_in)), dtype=np.uint16)
    volume = np.zeros((image_for_size.shape[0], image_for_size.shape[1], len(images_in)), dtype=np.float32)

    for index, image in tqdm(enumerate(images_in)):
        volume[:, :, index] = cv2.imread(folder_in + image, 2)

    writeNifti(path_out + name_out, volume)


stack_of_tiffs_to_nifti("outputs/", "./")





















