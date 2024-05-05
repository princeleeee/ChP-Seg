import os
import argparse
from utils.file_op import ImageData
from tools.skull_stripping import brain
from tools.ventricle_segmentation import get_ventricle
from tools.cp_segmentation import cp
from tools.put_cp_back import cp_way_back
from tensorflow import keras


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skull Stripping, lateral ventricle and choroid plexus segmentation pipeline')
    parser.add_argument("--input", type=str, metavar='File/Directory',help='File or directory to operate on, it can be \
        1) single Nifti file, 2) a dicom series folder, 3) a folder contatin many Nifti files, \
        4) .txt file each row contain the path of Nifti file.', default='demo/I812923.nii.gz')
    parser.add_argument("--output", metavar='SavePath', type=str, help='Cached files and results save path', default='results/')
    parser.add_argument("--job", metavar='Task', type=str, help='Options: "Full", "Stripping", "Ventricle", "ChoroidPlexus"', default='Full')
    parser.add_argument("--ven_weights", metavar='LateralVentricleWeights', type=str, default='weights/All_data_trainweights.200-0.05769.h5')
    parser.add_argument("--cp_weights", metavar='ChoroidPlexusWeights', type=str, default='weights/All_data_trainweights.184-0.96731.h5')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' # disable gpu

    # 1. generate .nii files and show summary info.
    dataset = ImageData(args.input, args.output)
    dataset.summary()

    # 2. run jobs.
    ven_weights = args.ven_weights
    cp_weights = args.cp_weights
    if args.job == "Full":
        nii_list = dataset.ids
        # skull stripping
        brain(nii_list, os.path.join(args.output, 'brain'))
        
        # ventricle segmentation
        brain_img_dir = os.path.join(args.output, 'brain', 'img')
        brain_img_list = [os.path.join(brain_img_dir, i) for i in os.listdir(brain_img_dir)]
        model = keras.models.load_model(ven_weights, compile=False)
        get_ventricle(brain_img_list, os.path.join(args.output, 'ventricle'), model)
        del model
        
        # choroid plexus segmentation
        ventricle_img_dir = os.path.join(args.output, 'ventricle', '1_img_crop')
        ventricle_img_list = [os.path.join(ventricle_img_dir, i) for i in os.listdir(ventricle_img_dir)]
        model = keras.models.load_model(cp_weights, compile=False)
        cp(ventricle_img_list, os.path.join(args.output, 'cp'), model)
        del model

    elif args.job == "Stripping":
        # skull stripping.
        brain(dataset.ids, os.path.join(args.output, 'brain'))
    
    elif args.job == "Ventricle":
        # ventricle segmentation, input -> brain image after skull stripping (img size: [160,200,160])
        model = keras.models.load_model(ven_weights, compile=False)
        get_ventricle(dataset.ids, os.path.join(args.output, 'ventricle'), model)
    
    elif args.job == "ChoroidPlexus":
        # ChoroidPlexus segmentation, input -> lateral ventricle area (img size: [96, 96, 80]).
        model = keras.models.load_model(cp_weights, compile=False)
        cp(dataset.ids, os.path.join(args.output, 'cp'), model)
    
    else:
        raise RuntimeError('unsupported task.')
    
    cp_way_back(args.output, args.output+'/file_collections.txt')