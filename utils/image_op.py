import numpy as np
import nibabel as nib
import os
import imageio
import sys
from .conform import is_conform, conform, check_affine_in_nifti
from scipy.ndimage import affine_transform
from matplotlib import pyplot as plt

def load_nifit(data_path):
    img = nib.load(data_path)
    tmp = np.squeeze(img.get_fdata()).astype(np.float32)
    return tmp


def save_nifit(data, filename, affine_matrix):
    # print(data.dtype)
    img = nib.Nifti1Image(data, affine=affine_matrix)
    nib.save(img, filename)


def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices), squeeze=False)
   for i, slice in enumerate(slices):
       axes[0, i].imshow(slice.T, cmap="gray", origin="lower")


def normlize_mean_std(tmp):
    tmp_std = np.std(tmp)
    tmp_mean = np.mean(tmp)
    # tmp = (tmp - tmp_mean) / tmp_std
    tmp = div0(tmp - tmp_mean, tmp_std)
    return tmp


def normlize_min_max(tmp):
    tmp_max = np.amax(tmp)
    tmp_min = np.amin(tmp)
    tmp = (tmp - tmp_min) / (tmp_max - tmp_min)
    return tmp


def div0(a, b):
    if b == 0:
        c = np.zeros_like(a)
    else:
        c = a / b
    return c


def crop_pad3D(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)
    start_pos = start_pos.astype(int)
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    return y

def gif_build(images_dir: str, savepath: str):
    """
    images_dir: directory contains a series of images to build a .gif file. The images in this dir must be named increasely.
    savepath: path of genereated .gif file.
    """
    filenames = os.listdir(images_dir)
    filenames.sort()
    with imageio.get_writer(savepath, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(images_dir, filename))
            writer.append_data(image)

# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm isotropic voxels
def load_and_conform_image(img_filename, interpol=1, logger=None):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)

    if not is_conform(orig):

        if logger is not None:
            logger.info('Conforming image to UCHAR, RAS orientation, and 1mm isotropic voxels')
        else:
            print('Conforming image to UCHAR, RAS orientation, and 1mm isotropic voxels')

        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # conform
        orig = conform(orig, interpol)

    return orig

def get_bounds(shape, affine):
    """Return the world-space bounds occupied by an array given an affine.

    The coordinates returned correspond to the **center** of the corner voxels.

    Parameters
    ----------
    shape : tuple
        shape of the array. Must have 3 integer values.

    affine : numpy.ndarray
        affine giving the linear transformation between voxel coordinates
        and world-space coordinates.

    Returns
    -------
    coord : list of tuples
        coord[i] is a 2-tuple giving minimal and maximal coordinates along
        i-th axis.

    """
    adim, bdim, cdim = shape
    adim -= 1
    bdim -= 1
    cdim -= 1
    # form a collection of vectors for each 8 corners of the box
    box = np.array([[0.,   0,    0,    1],
                    [adim, 0,    0,    1],
                    [0,    bdim, 0,    1],
                    [0,    0,    cdim, 1],
                    [adim, bdim, 0,    1],
                    [adim, 0,    cdim, 1],
                    [0,    bdim, cdim, 1],
                    [adim, bdim, cdim, 1]]).T
    box = np.dot(affine, box)[:3]
    return list(zip(box.min(axis=-1), box.max(axis=-1)))

class BoundingBoxError(ValueError):
    """This error is raised when a resampling transformation is
    incompatible with the given data.

    This can happen, for example, if the field of view of a target affine
    matrix does not contain any of the original data."""
    pass

def to_matrix_vector(transform):
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector

def orig2ras_isotropic(orig_img, outshape=None, mask=False):
    # Now only support 3D images only. 1 mm isotropic. RAS+
    target_affine = np.eye(4)
    transform_affine = np.linalg.inv(target_affine).dot(orig_img.affine)

    # compute the offset component of target affine matrix.
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(
        orig_img.get_fdata().shape[:3], transform_affine)
    offset = target_affine[:3, :3].dot([xmin, ymin, zmin])
    target_affine[:3, 3] = offset
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = (
            (0, xmax - xmin), (0, ymax - ymin), (0, zmax - zmin))

    if outshape is None:
        outshape = (int(np.ceil(xmax)) + 1,
                        int(np.ceil(ymax)) + 1,
                        int(np.ceil(zmax)) + 1)
    # Check whether transformed data is actually within the FOV
    # of the target affine
    if xmax < 0 or ymax < 0 or zmax < 0:
        raise BoundingBoxError("The field of view given "
                               "by the target affine does "
                               "not contain any of the data")
    
    transform_affine = np.linalg.inv(orig_img.affine).dot(target_affine)
    A, b = to_matrix_vector(transform_affine)

    orig_img_array = np.squeeze(orig_img.get_fdata())
    if not mask:
        new_data = affine_transform(orig_img_array, A, offset=b, output_shape=outshape, order=3)  # cubic for image
    else:
        new_data = affine_transform(orig_img_array, A, offset=b, output_shape=outshape, order=0)  # nearst for mask
    return nib.Nifti1Image(new_data, affine=target_affine)

def inverse_resample(img_resampled_path, orig_img_path, mask=False):
    """
    Inverse the image resampled by funcitons(orig2ras_isotropic or nilearn.image.resample_img).
    :param img_resampled_path: a transform image.(example: an image generate by conform function in FastSurfer(256^3, LIA orientation)...)
    :param orig_img_path: original image without any processing.
    :param int mask: mask or image.
    """
    img_resampled = nib.load(img_resampled_path)
    img_orig = nib.load(orig_img_path)
    target_affine = img_orig.affine
    outshape = img_orig.shape
    new_header = img_orig.header

    if img_orig.ndim ==4 and 1 in outshape:
        print(f'Warning: Original image is 4 dimensions: {outshape}')
        outshape = np.squeeze(img_orig.get_fdata()).shape

    transform_affine = np.linalg.inv(img_resampled.affine).dot(target_affine)
    A, b = to_matrix_vector(transform_affine)
    img_resampled_array = np.squeeze(img_resampled.get_fdata())

    if not mask:
        new_data = affine_transform(img_resampled_array, A, offset=b, output_shape=outshape, order=3)  # cubic for image
    else:
        new_data = affine_transform(img_resampled_array, A, offset=b, output_shape=outshape, order=0)  # nearst for mask
    
    # avoid some dtype convert error. https://neurostars.org/t/how-to-change-the-datatype-of-a-niftiimage-and-save-it-with-that-datatype/4809/3
    new_header.set_data_dtype(np.float32)
    return nib.Nifti1Image(new_data, affine=target_affine, header=new_header)

def change_affine(affine, pad_range, crop_range):
    affine[:3, 3] -= [pad_range[0][0], pad_range[1][0], pad_range[2][0]]
    affine[:3, 3] += [crop_range[0], crop_range[2], crop_range[4]]
    return affine

def mgh2nii(mghfpath, dtype=None):
    # convert MGH format to .nii
    mgh = nib.load(mghfpath)
    img_array = mgh.get_fdata()
    if dtype is not None:
        img_array = img_array.astype(dtype)
    affine = mgh.affine
    nii = nib.Nifti1Image(img_array, affine=affine)
    return nii

def single_imshow(I, name=None, vmin=None, vmax=None, dpi=96, colorbar=False, figsize=(6.4,4.8), axis_off=False):
    plt.figure(figsize=figsize)
    plt.imshow(I, cmap='gray', vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar()
    if axis_off:
        plt.axis('off')
    plt.tight_layout()

    if name:
        plt.savefig(name, dpi=dpi)
        plt.close()
        plt.clf
    else:
        plt.show()
        
def imshow(I, name=None, vmin=None, vmax=None, dpi=96, colorbar=False, figsize=(6.4,4.8), axis_off=False):
    # the I[0] slice dimension
    # print(I.shape)
    if len(I.shape) == 2:
        N1 = 1
        N2 = 1
        image_n1 = I.shape[0]
        image_n2 = I.shape[1]
        I = I[np.newaxis,:,:]
    elif len(I.shape) == 3:
        # print('3D')
        image_n1 = I.shape[1]
        image_n2 = I.shape[2]
        N2 = np.ceil(np.sqrt(I.shape[0])).astype(np.int16)
        N1 = np.ceil(I.shape[0]/N2).astype(np.int16)
        tmp = np.zeros([N1*N2, image_n1, image_n2])
        tmp[:I.shape[0]]+= I
        I =tmp
    elif len(I.shape) == 4:
        N2 = I.shape[0] 
        N1 = I.shape[1]
        image_n1 = I.shape[2]
        image_n2 = I.shape[3] 
        I = I.reshape([N1*N2, image_n1, image_n2])
    # print(I.shape)
    F = np.zeros((image_n1*N2, image_n2*N1))
    frame = 0
    # print('N1:', N1)
    # print('N2:', N2)
    for n2 in range(N2):
        for n1 in range(N1):
            # print(N1, N2, frame)
            F[n2*image_n1:(n2+1)*image_n1, n1*image_n2:(n1+1)*image_n2] = np.squeeze(I[frame])
            frame += 1
    single_imshow(F, name=name, vmin=vmin, vmax=vmax, dpi=dpi, colorbar=colorbar, figsize=figsize, axis_off=axis_off)

    return F

if __name__ == "__main__":
    print("Hello world.")