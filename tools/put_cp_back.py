import os
import re
import argparse
import numpy as np
import nibabel as nib
from skimage.morphology import remove_small_objects
from utils.image_op import inverse_resample
from utils.file_op import mkdirs

def get_crop_range(croptxt):
    refer_dict = {}
    with open(croptxt, 'r') as f:
        alllines = f.readlines()
    for line in alllines:
        line = line.strip()
        if not line == '':
            key  = line.split(';')[0].split('/')[-1].split('.nii')[0]
            padding_range = line.split(';')[1].split(':')[-1]
            crop_range = line.split(';')[2].split(':')[-1]
            refer_dict[key] = [eval(padding_range), eval(crop_range)]
    return refer_dict

def affine_reverse(affine, pad_range, crop_range):
    affine[:3, 3] -= [crop_range[0], crop_range[2], crop_range[4]]
    affine[:3, 3] += [pad_range[0][0], pad_range[1][0], pad_range[2][0]]
    return affine

def put_back_new(imgarray, affine, paddingrange, croprange, orig_shape=[160, 200, 160], id=None):
    # crop range: first value included, last value excluded.
    # paddingrange: padding range during the ventricle segmentation and crop stage.
    # croprange: corp range during the ventricle segmentation and crop stage(orig image has been padded).
    
    if paddingrange == [(0, 0), (0, 0), (0, 0)]:
        fillrange = ((croprange[0], orig_shape[0]-croprange[1]), (croprange[2], orig_shape[1]-croprange[3]), 
                     (croprange[4], orig_shape[2]-croprange[5]))
        img_array_padded = np.pad(imgarray, fillrange, 'constant', constant_values=0)
        new_affine = affine_reverse(affine, paddingrange, croprange)
    else:
        new_affine = affine_reverse(affine, paddingrange, croprange)
        delta = [paddingrange[0][0], paddingrange[0][0]+paddingrange[0][1],
                paddingrange[1][0], paddingrange[1][0]+paddingrange[1][1],
                paddingrange[2][0], paddingrange[2][0]+paddingrange[2][1]]
        croprange_before_padding = np.array(croprange) - np.array(delta)
        croprange_before_padding[np.where(croprange_before_padding<0)]=0

        fillrange = ((croprange_before_padding[0], orig_shape[0]-croprange_before_padding[1]),
                     (croprange_before_padding[2], orig_shape[1]-croprange_before_padding[3]), 
                     (croprange_before_padding[4], orig_shape[2]-croprange_before_padding[5]))

        # first remove the padding area.
        img_array_before_padding = imgarray[paddingrange[0][0]:imgarray.shape[0]-paddingrange[0][1],
                                            paddingrange[1][0]:imgarray.shape[1]-paddingrange[1][1],
                                            paddingrange[2][0]:imgarray.shape[2]-paddingrange[2][1]]
        img_array_padded = np.pad(img_array_before_padding, fillrange, 'constant', constant_values=0)
        print('-'*100)
        print(id)
        print('', paddingrange)
        print('', croprange)
        print('', croprange_before_padding)
        print('', fillrange)
        print('-'*100)
        
    return img_array_padded, new_affine

def cp_way_back(pipeline_path, original_images_list):
    brain_crop_range = get_crop_range(pipeline_path + "brain/resample/crop_range.txt")
    ventricle_crop_range = get_crop_range(pipeline_path + "ventricle/crop_range.txt")
    with open(original_images_list, 'r') as f:
        orig_images_file_text = f.read()
    
    ven_dir = pipeline_path + 'ventricle/0_mask/'
    ven_resampledT1_savedir = pipeline_path + 'ventricle/2_resampledT1_space/'
    ven_origT1_savedir = pipeline_path + 'ventricle/3_orig_T1_space/'

    cp_dir = pipeline_path + 'cp/0_mask'
    cp_refine_savedir = pipeline_path + 'cp/1_mask_refine'
    cp_resampledT1_savedir = pipeline_path + 'cp/2_resampledT1_space'
    cp_origT1_savedir = pipeline_path + 'cp/3_orig_T1_space'

    brain_resample_img_dir = pipeline_path + 'brain/resample'
    brain_resample_inverse_savedir = pipeline_path + "brain/resample_inverse"
    
    mkdirs(cp_refine_savedir)
    mkdirs(cp_resampledT1_savedir)
    mkdirs(cp_origT1_savedir)
    mkdirs(ven_resampledT1_savedir)
    mkdirs(ven_origT1_savedir)
    mkdirs(brain_resample_inverse_savedir)

    for nii in os.listdir(cp_dir):
        # refine cp segmentation results.
        cp = nib.load(os.path.join(cp_dir, nii))
        cp_data = cp.get_fdata()
        cp_data_refine = remove_small_objects(cp_data>0, min_size=30, connectivity=3)
        cp_refine_img = nib.Nifti1Image(cp_data_refine.astype(np.int8), affine=cp.affine)
        nib.save(cp_refine_img, os.path.join(cp_refine_savedir, nii))

        # cp restored to brain image 
        cp_in_brain_array, cp_in_brain_affine = put_back_new(cp_refine_img.get_fdata(), cp_refine_img.affine.copy(),\
             ventricle_crop_range[nii.split('.nii')[0]][0], ventricle_crop_range[nii.split('.nii')[0]][1], orig_shape=[160, 200, 160])
        assert cp_in_brain_array.shape == (160, 200, 160)

        brain_resample_img = nib.load(os.path.join(brain_resample_img_dir, nii))    # resampled RAS 1mm^3 T1w image.
        
        # cp restored to resampled t1 image size
        cp_restore_array, cp_restore_affine = put_back_new(cp_in_brain_array, cp_in_brain_affine,\
             brain_crop_range[nii.split('.nii')[0]][0], brain_crop_range[nii.split('.nii')[0]][1], orig_shape=brain_resample_img.shape, id=os.path.join(cp_dir, nii))
        assert cp_restore_array.shape == brain_resample_img.shape
        cp_restore_img = nib.Nifti1Image(cp_restore_array, cp_restore_affine)
        nib.save(cp_restore_img, os.path.join(cp_resampledT1_savedir, nii))

        # ventricle restored to resampled t1 image size
        ventricle_mask_nifti = nib.load(ven_dir+nii)
        ven_restore_array, ven_restore_affine = put_back_new(ventricle_mask_nifti.get_fdata(), ventricle_mask_nifti.affine.copy(),\
             brain_crop_range[nii.split('.nii')[0]][0], brain_crop_range[nii.split('.nii')[0]][1], orig_shape=brain_resample_img.shape, id=os.path.join(ven_dir, nii))
        assert ven_restore_array.shape == brain_resample_img.shape
        ven_restore_img = nib.Nifti1Image(ven_restore_array, ven_restore_affine)
        nib.save(ven_restore_img, os.path.join(ven_resampledT1_savedir, nii))

        # inverse resampled cp, ven and t1 to original coordinate space.
        img_orig_path = re.findall(f".*{nii.split('.nii')[0]}.nii.*", orig_images_file_text) # original T1w image.
        assert len(img_orig_path) == 1
        img_orig_path = img_orig_path[0]
        cp2origT1_space = inverse_resample(os.path.join(cp_resampledT1_savedir, nii), img_orig_path, mask=True)
        ven2origT1_space = inverse_resample(os.path.join(ven_resampledT1_savedir, nii), img_orig_path, mask=True)
        resampledT1_to_original_space = inverse_resample(os.path.join(brain_resample_img_dir, nii), img_orig_path, mask=False)

        # save original T1w space cp, ventricle segmentation results and T1w.
        nib.save(cp2origT1_space, os.path.join(cp_origT1_savedir, nii))
        nib.save(ven2origT1_space, os.path.join(ven_origT1_savedir, nii))
        nib.save(resampledT1_to_original_space, os.path.join(brain_resample_inverse_savedir, nii))
    
    return