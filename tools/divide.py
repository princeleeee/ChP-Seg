import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from scipy.stats import zscore
import os
import sys
import logging

def remove_outliers(points, logger: logging.Logger, threshold=5):
    z_scores = np.abs(zscore(points, axis=0))
    outliers = points[(z_scores >= threshold).any(axis=1)]
    filtered_points = points[(z_scores < threshold).all(axis=1)]
    
    # print("Removed outliers:")
    for outlier in outliers:
        # print(outlier)
        logger.warning(f"Removed outlier: {outlier}")
    
    if outliers.size == 0:
        return filtered_points, False
    else:
        return filtered_points, True

def kmeans_split_choroid_plexus_3d(mask, logger: logging.Logger):
    depth, height, width = mask.shape
    
    points = np.column_stack(np.where(mask == 1))
    
    filtered_points, _ = remove_outliers(points, logger)

    if filtered_points.size != 0:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(filtered_points)
        labels = kmeans.labels_
        
        new_mask = np.zeros((depth, height, width), dtype=np.uint8)
        
        for i, point in enumerate(filtered_points):
            if labels[i] == 0:
                new_mask[tuple(point)] = 1
            else:
                new_mask[tuple(point)] = 2

    else:
        new_mask = np.zeros((depth, height, width), dtype=np.uint8)
        logger.warning("No foreground voxels found in the mask.")
    
    return new_mask


def divide_directory(input_dir, output_dir, logger: logging.Logger):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"Error: The directory '{output_dir}' already exists.")
        logger.error(f"Error: The directory '{output_dir}' already exists.")
        sys.exit(1)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"divided_{filename}")
            
            nii = nib.load(input_path)
            mask = nii.get_fdata()
            
            # print(f"Processing: {input_path}")
            logger.info(f"Processing: {input_path}")
            new_mask = kmeans_split_choroid_plexus_3d(mask, logger)
            new_nii = nib.Nifti1Image(new_mask, nii.affine)
            nib.save(new_nii, output_path)
            # print(f"saved: {output_path}")
            logger.info(f"Saved: {output_path}")

def points_check(input_dir, logger: logging.Logger):
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, filename)
            nii = nib.load(input_path)
            mask = nii.get_fdata()
            if (mask == 0).all():
                logger.info(f"Empty mask: {input_path}")

if __name__ == '__main__':
    pass