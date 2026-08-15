import SimpleITK as sitk
from radiomics import shape
import os
import nibabel as nib
import numpy as np
import logging
import pandas as pd
from tools.divide import divide_directory

def ensure_logs_dir(logs_dir):
    os.makedirs(logs_dir, exist_ok=True)

def get_shape_features(input_dir):
    inf = []
    print(input_dir)
    for filename in os.listdir(input_dir):
        # print(filename)
        if filename.endswith(".nii.gz"):
            # print(filename)
            mask = sitk.ReadImage(os.path.join(input_dir, filename))

            for i in range(1,3):
                mask_label = sitk.BinaryThreshold(mask, i, i, 1, 0)
                shapeFeatures = shape.RadiomicsShape(mask_label, mask_label)
                shapeFeatures.enableAllFeatures()
                shapeFeatures.enableFeatureByName('Compactness1')
                shapeFeatures.enableFeatureByName('Compactness2')
                shapeFeatures.enableFeatureByName('SphericalDisproportion')

                result = shapeFeatures.execute()
                result = {
                    'file_name': filename.strip('divided_'),
                    'chp_component_label': i,
                    **result
                }
                inf.append(result)

    return inf

def extract_pipeline_chp_features(pipeline_root, csv_save_path):
    logs_dir = os.path.join(csv_save_path, 'logs')
    ensure_logs_dir(logs_dir)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('normal people choroid plexus two sides identification.')
    logger.setLevel(logging.INFO)

    file_handler1 = logging.FileHandler(os.path.join(logs_dir, 'dir_divide_normal.log'))
    file_handler1.setFormatter(formatter)
    logger.addHandler(file_handler1)

    try:
        shape_features = []

        input_directory = os.path.join(pipeline_root, 'cp', '0_mask')
        output_directory = os.path.join(pipeline_root, 'cp', 'divided_0_mask')

        divide_directory(input_directory, output_directory, logger)

        result = get_shape_features(output_directory)
        shape_features.extend(result)

        df = pd.DataFrame(shape_features)
        df.to_csv(os.path.join(csv_save_path, 'chp_3D_shape_features.csv'), index=False)
    finally:
        logger.removeHandler(file_handler1)
        file_handler1.close()
