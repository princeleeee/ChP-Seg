import numpy as np
import nibabel as nib
import os
from utils.image_op import normlize_mean_std, save_nifit
import datetime
from utils.file_op import mkdirs

def cp(ventricle_list, output_dir, model):
    mask_dir = os.path.join(output_dir, '0_mask')
    mkdirs(mask_dir)

    starttime = datetime.datetime.now()

    for file_path in ventricle_list:
        img = nib.load(file_path)
        img_array = img.get_fdata().astype(np.float32)
        
        x = normlize_mean_std(img_array)
        x = np.expand_dims(x, axis=(0,4))

        prediction = model.predict(x)  # dimension: [batch, x, y, z, intensity]
        mask = (prediction > 0.5).squeeze().astype(np.int8)
        save_nifit(mask, os.path.join(mask_dir, file_path.split('/')[-1]), img.affine)

    now = datetime.datetime.now()
    print('total choroid plexus segmentation running time:{}s, start time is {}, finish time is {}'.format(((now-starttime).seconds), starttime, now))