import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
from deepbrain import Extractor
import os.path
import pandas as pd
pd.options.mode.chained_assignment = None
from skimage.morphology import remove_small_objects
import multiprocessing
import datetime
from functools import partial
from utils.file_op import mkdirs
from utils.image_op import orig2ras_isotropic, change_affine

def view3plane(I, figurename=None, pos=None):
    if not figurename:
        figurename = 'tmp'
    if not pos:
        image_size = I.shape
        pos = np.asarray(image_size)//2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(I[pos[0], :, :], cmap='gray'), ax[0].set_ylabel('dim=1'), ax[0].set_xlabel('dim=2'),
    ax[1].imshow(I[:, pos[1], :], cmap='gray'), ax[1].set_ylabel('dim=0'), ax[1].set_xlabel('dim=2'),
    ax[2].imshow(I[:, :, pos[2]], cmap='gray'), ax[2].set_ylabel('dim=0'), ax[2].set_xlabel('dim=1'), 
    plt.tight_layout()
    plt.savefig(figurename+'.png')
    plt.close(fig)
    
def round2(x):
    'if the length of x is odd, then add one more elemenet'
    if len(x)%2:
        y = x.item(-1)+2
    else:
        y = x.item(-1)+1
    return [x.item(0), y]

def crop_center(img_array, target_size, mask_array=None):
    'crop 3D volume to the target size around the content center'
    x, y, z = img_array.shape

    # get center point of 3D image (center0, center1, center2)
    I0 = np.sum(img_array, axis=(1, 2))
    index0 = round2(np.argwhere(I0>0))
    center0 = (index0[0]+index0[1])//2

    I1 = np.sum(img_array, axis=(0, 2))
    index1 = round2(np.argwhere(I1>0))
    center1 = (index1[0]+index1[1])//2

    I2 = np.sum(img_array, axis=(0, 1))
    index2 = round2(np.argwhere(I2>0))
    center2 = (index2[0]+index2[1])//2

    # range
    image_index_range = ((0, x-1), (0, y-1), (0, z-1))
    crop_index_range = ((center0 - target_size[0]//2, center0+target_size[0]//2),
          (center1 - target_size[1]//2, center1+target_size[1]//2),
          (center2 - target_size[2]//2, center2+target_size[2]//2))
    padding_range = []
    
    # calculate padding range
    padded_flag = False
    for index_before, index_after in zip(image_index_range, crop_index_range):
        new_index = []
        if index_after[0] < 0:
            new_index.append(abs(index_after[0]))
            padded_flag = True
        else:
            new_index.append(0)
        
        if index_after[1] > index_before[1]:
            new_index.append(abs(index_after[1] - index_before[1]))
            padded_flag = True
        else:
            new_index.append(0)

        padding_range.append(tuple(new_index))
    
    center0 += padding_range[0][0]
    center1 += padding_range[1][0]
    center2 += padding_range[2][0]

    img_array_padded = np.pad(img_array, tuple(padding_range), 'constant', constant_values=0)
    img_cropped = img_array_padded[center0 - target_size[0]//2:center0+target_size[0]//2,
          center1 - target_size[1]//2:center1+target_size[1]//2,
          center2 - target_size[2]//2:center2+target_size[2]//2]
    
    crop_range_padded = [center0 - target_size[0]//2, center0+target_size[0]//2,
            center1 - target_size[1]//2, center1+target_size[1]//2,
            center2 - target_size[2]//2, center2+target_size[2]//2]             
    
    if mask_array is not None:
        mask_array_padded = np.pad(mask_array, tuple(padding_range), 'constant', constant_values=0)
        mask_cropped = mask_array_padded[center0 - target_size[0]//2:center0+target_size[0]//2,
            center1 - target_size[1]//2:center1+target_size[1]//2,
            center2 - target_size[2]//2:center2+target_size[2]//2]
        return img_cropped, padding_range, padded_flag, mask_cropped, crop_range_padded

    return img_cropped, padding_range, padded_flag, crop_range_padded

def core(nii_filepath, outputpath):
    brain_img_dir, brain_mask_dir, check_dir, err_dir, resample_img_dir = outputpath

    imageuid = nii_filepath.split('/')[-1].split('.')[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable gpu
    ext = Extractor()
    # print(imageuid, nii_filepath, flush=True)
    try:
        source_img = nib.load(nii_filepath)
        upsample_img = orig2ras_isotropic(source_img) # RAS+ 1mm^3
        nib.save(upsample_img, os.path.join(resample_img_dir, imageuid+'.nii.gz'))
        # extract brain https://github.com/rockstreamguy/deepbrain
        img = upsample_img.get_fdata()

        mask = ext.run(img) > 0.5
        brain = img*mask
        no_small_mask = remove_small_objects(brain>0, 50000)
        brain = img*mask*no_small_mask

        crop_array, padding_range, pad_flag, mask_crop_array, crop_range_padded = crop_center(brain,[160,200,160], mask.astype(np.uint8))
        
        with open(os.path.join(resample_img_dir, 'crop_range.txt'), 'a') as f:
                f.write("\n")
                f.write(f'File:{nii_filepath};Padding range:{padding_range};Crop range after padding:{crop_range_padded}')
        
        new_affine = upsample_img.affine
        new_affine = change_affine(new_affine, padding_range, crop_range_padded)
        if pad_flag:
            print('Doing Skull stripping | padding range for crop center op: ', padding_range, '<--- '+nii_filepath)

        # error cases: brain was cropped
        if(np.sum(crop_array[0])+np.sum(crop_array[-1])+np.sum(crop_array[:,0,:])+np.sum(crop_array[:,-1,:])+np.sum(crop_array[:,:,0])+np.sum(crop_array[:,:,-1]))>0: 
            view3plane(crop_array, figurename=os.path.join(err_dir, imageuid))
        else:
            view3plane(crop_array, figurename=os.path.join(check_dir, imageuid))

        brain_crop_img = nib.Nifti1Image(crop_array, affine=new_affine)
        brain_crop_mask = nib.Nifti1Image(mask_crop_array, affine=new_affine)
        nib.save(brain_crop_img, os.path.join(brain_img_dir, imageuid+'.nii.gz'))
        nib.save(brain_crop_mask, os.path.join(brain_mask_dir, imageuid+'.nii.gz'))

        del img, upsample_img, mask, brain
    except:
        print('Doing Skull stripping | !!! unable to read this data: ', nii_filepath)

def brain(nii_file, output_dir):
    outputdirs = [os.path.join(output_dir, 'img'), os.path.join(output_dir, 'mask'), \
        os.path.join(output_dir, 'check'), os.path.join(output_dir, 'err'), os.path.join(output_dir, 'resample')]
    
    for i in outputdirs:
        mkdirs(i)
    starttime = datetime.datetime.now()
    with multiprocessing.Pool(8) as p:
        p.map(partial(core, outputpath=outputdirs), nii_file)
    current_time = datetime.datetime.now()
    print('Total skull stripping running time:{}s, start time is {}, finish time is {}'.format(((current_time-starttime).seconds), starttime, current_time))