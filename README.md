# A pipeline of choroid plexus (ChP) segmentation for T1w MRI

## Introduction
In this repository, we offer two methods to execute the ChP segmentation pipeline. One method involves directly executing the Python code, while the other method utilizes Docker.

## Python code
In this way, the following input forms are accepeted:
1. Single `.nii` or `.nii.gz` file
2. A folder containing `.dcm` series
3. A folder containing multiple Nifti files
4. A `.txt` file where each row contains the Nifti file path
```bash
python pipeline.py --input demo/I812923.nii.gz
```

## Docker
```bash
docker run -v $WEIGHTS_PATH_ON_HOST:/app/weights -v $OUTPUT_FOLDER_ON_HOST:/app/results -it --rm chp-seg bash
```

## Citation
If our code and work helps you, please cite our paper.