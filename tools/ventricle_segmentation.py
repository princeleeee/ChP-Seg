import numpy as np
import nibabel as nib
import os
import datetime
from utils.file_op import mkdirs
from utils.image_op import normlize_mean_std, save_nifit, change_affine

def crop_center(image_array, mask_array, crop_size=None, center_mode = 'average'):
    """
    center_mode: 'cube' or 'average'; 'cube' means the Smallest circumscribed cube arround mask 
    """
    image_index_range = ((0, image_array.shape[0]-1), (0, image_array.shape[1]-1), (0, image_array.shape[2]-1))

    # non-zero range of mask array. cube
    mask_x_range, mask_y_range, mask_z_range = np.where(mask_array == 1)
    mask_range = [(mask_x_range.min(), mask_x_range.max()), (mask_y_range.min(), mask_y_range.max()), (mask_z_range.min(), mask_z_range.max())]
    
    if not crop_size:  # Only extract the image under mask.
        image_cropped = image_array[mask_range[0][0]:mask_range[0][1], mask_range[1][0]:mask_range[1][1], mask_range[2][0]:mask_range[2][1]]
        mask_cropped = mask_array[mask_range[0][0]:mask_range[0][1], mask_range[1][0]:mask_range[1][1], mask_range[2][0]:mask_range[2][1]]
    else:  # handle the occassion which cropped image/mask out of the original image/mask range.
        if center_mode == 'minimum-cube':
            crop_center = [round((i[0]+i[1])/2) for i in mask_range]
        elif center_mode == 'average':
            crop_center = [round(mask_x_range.mean()), round(mask_y_range.mean()), round(mask_z_range.mean())]
        crop_range = [(crop_center[0]-crop_size[0]/2, crop_center[0]+crop_size[0]/2), 
                        (crop_center[1]-crop_size[1]/2, crop_center[1]+crop_size[1]/2),
                        (crop_center[2]-crop_size[2]/2, crop_center[2]+crop_size[2]/2)]
        
        padded_flag = False
        padding_range = []
        for index_before, index_after in zip(image_index_range, crop_range):
            padded_num = []
            if index_after[0] < 0:
                padded_num.append(int(abs(index_after[0])))
                padded_flag = True
            else:
                padded_num.append(0)
            
            if index_after[1] > index_before[1]:
                padded_num.append(int(abs(index_after[1] - index_before[1])))
                padded_flag = True
            else:
                padded_num.append(0)

            padding_range.append(tuple(padded_num))

        crop_center[0] += padding_range[0][0]
        crop_center[1] += padding_range[1][0]
        crop_center[2] += padding_range[2][0]

        image_array_padded = np.pad(image_array, tuple(padding_range), 'constant', constant_values=0)
        mask_array_padded = np.pad(mask_array, tuple(padding_range), 'constant', constant_values=0)
        crop_range_padded =  [crop_center[0] - crop_size[0]//2, crop_center[0]+crop_size[0]//2,
            crop_center[1] - crop_size[1]//2, crop_center[1]+crop_size[1]//2,
            crop_center[2] - crop_size[2]//2, crop_center[2]+crop_size[2]//2]
        crop_range_padded = [int(i) for i in crop_range_padded]
        image_cropped = image_array_padded[crop_range_padded[0]:crop_range_padded[1], crop_range_padded[2]:crop_range_padded[3], crop_range_padded[4]:crop_range_padded[5]]
        mask_cropped = mask_array_padded[crop_range_padded[0]:crop_range_padded[1], crop_range_padded[2]:crop_range_padded[3], crop_range_padded[4]:crop_range_padded[5]]        
    
    return image_cropped, mask_cropped, padding_range, crop_range_padded

def get_ventricle(brain_list, output_dir, model, crop=True):
    ventricle_mask_dir = os.path.join(output_dir, '0_mask')
    ventricle_crop_dir = os.path.join(output_dir, '1_img_crop')
    ventricle_mask_crop_dir = os.path.join(output_dir, '1_mask_crop')
    for i in [ventricle_mask_dir, ventricle_crop_dir, ventricle_mask_crop_dir]:
        mkdirs(i) 

    starttime = datetime.datetime.now()

    for file_path in brain_list:
        img = nib.load(file_path)
        img_array = img.get_fdata().astype(np.float32)
        
        x = normlize_mean_std(img_array)
        x = np.expand_dims(x, axis=(0,4))

        prediction = model.predict(x)  # dimension: [batch, x, y, z, intensity]
        mask = (prediction > 0.5).squeeze().astype(np.int8)
        save_nifit(mask, os.path.join(ventricle_mask_dir, file_path.split('/')[-1]), img.affine)

        if crop:
            ventricle_crop, mask_crop, padding_range, crop_range_padded\
                 = crop_center(img_array, mask, crop_size=[96, 96, 80])
            if padding_range != [(0, 0), (0, 0), (0, 0)]:
                print(file_path, padding_range)
            
            new_affine = change_affine(img.affine, padding_range, crop_range_padded)
            save_nifit(ventricle_crop, os.path.join(ventricle_crop_dir, file_path.split('/')[-1]), new_affine)
            save_nifit(mask_crop, os.path.join(ventricle_mask_crop_dir, file_path.split('/')[-1]), new_affine)
            
            with open(os.path.join(output_dir, 'crop_range.txt'), 'a') as f:
                f.write("\n")
                f.write(f'File:{file_path};Padding range:{padding_range};Crop range after padding:{crop_range_padded}')

    current_time = datetime.datetime.now()
    print('total ventricle segmentation time:{}s, start time is {}, finish time is {}'\
        .format(((current_time-starttime).seconds), starttime, current_time))

