import os
import json
import numpy as np
import SimpleITK as sitk
import time

def series2dict(pdseries):
    # convert single pandas series to dict
    d = {column:pdseries.get(column) for column in pdseries.index}
    return d

class NpEncoder(json.JSONEncoder):
    # useful when save .json 
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def file_dir(dpath):
    # get files and dirs in the given path.
    total = os.listdir(dpath)
    files = []
    dirs = []
    for item in total:
        if os.path.isdir(os.path.join(dpath, item)):
            dirs.append(item)
        elif os.path.isfile(os.path.join(dpath, item)):
            files.append(item)
        else:
            RuntimeError()
    return files, dirs

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def dcmseries2nii(input_directory, output_file):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_file)

def nii2dcmseries(nii_path, output_directory):
    # reference to https://simpleitk.readthedocs.io/en/next/Examples/DicomSeriesFromArray/Documentation.html, and
    # https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/03_Image_Details.ipynb
    mkdirs(output_directory)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.LoadPrivateTagsOn()
    reader.SetFileName(nii_path)
    image = reader.Execute()

    tags_to_copy = ["0010|0010", # Patient Name
                "0010|0020", # Patient ID
                "0010|0030", # Patient Birth Date
                "0020|000D", # Study Instance UID, for machine consumption
                "0020|0010", # Study ID, for human consumption
                "0008|0020", # Study Date
                "0008|0030", # Study Time
                "0008|0050", # Accession Number
                "0008|0060"  # Modality
                ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = image.GetDirection()
    series_tag_values = [(k, reader.GetMetaData(k)) for k in tags_to_copy if reader.HasMetaDataKey(k)] + \
                    [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|000D", "1.2.826.0.1.3680043.2.1125."+modification_date+".2"+modification_time), # Study Instance UID, for machine consumption
                    ("0020|0010", "1.2.826.0.1.3680043.2.1125."+modification_date+".3"+modification_time), # Study ID, for human consumption
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7]))))]
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    image_array_tmp = sitk.GetArrayFromImage(image).astype(np.int16)
    image_tmp = sitk.GetImageFromArray(image_array_tmp)
    image_tmp.CopyInformation(image)

    for i in range(image.GetDepth()):
        image_slice = image_tmp[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(output_directory, str(i+1).zfill(3)+'.dcm'))
        writer.Execute(image_slice)
    return

class ImageData():
    def __init__(self, path, save_dir=None):
        self.path = path
        self.save_dir = save_dir
        mkdirs(self.save_dir)
        self.convert_flag = False
        self.ids_orig = list()
        self.ids = list()
        if self.path.endswith('.nii') or self.path.endswith('.nii.gz'):    # .nii.gz or .nii file
            self.ids.append(self.path)
            self.convert_flag = True
        elif self.path.endswith('.txt'):
            with open(path, 'r') as f:
                nii_list = f.readlines()
                nii_list = [line.strip() for line in nii_list]
                self.ids = nii_list
                self.convert_flag = True
        else:
            for root, dirs, files in  os.walk(self.path):
                if len(dirs) == 0 and '.dcm' in files[0]:   # folder contatins many .dcm files.
                    self.ids_orig.append(root)
                elif len(dirs) == 0 and '.nii' in files[0]:   # folder contains .nii file(s).
                    self.ids = [os.path.join(root, file) for file in files]
                    self.convert_flag = True
                elif len(dirs) != 0 and len(files) == 0:   # folder contains dcm subfolder(s).
                    self.ids_orig = [os.path.join(root, dir) for dir in dirs]
                else:
                    raise RuntimeError("Don't support this kind of data filt structure currently.")
                break
        self.convert2nii()
        with open(f'{self.save_dir}/file_collections.txt', 'w') as f:
            for id in self.ids:
                f.write(f"{id}\n")

    def convert2nii(self):
        if self.convert_flag is False:
            # add tqdm
            for id in self.ids_orig:
                output_name = id.split('/')[-1]+'.nii.gz'
                convert_save_folder = self.save_dir + 'convert_dicom_to_nii'
                mkdirs(convert_save_folder)
                dcmseries2nii(id, os.path.join(convert_save_folder, output_name))
                self.ids.append(os.path.join(convert_save_folder, output_name))
        else:
            print("Don't need to be converted to .nii format.")
        return
    
    def summary(self):
        print("*"*30, 'Dataset Summary Information', '*'*30)
        print(f"Source data path: {self.path}")
        print(f"Total {len(self.ids)} .nii files.")
        print("-"*89)
        return

    def showimage(self, index):
        return
    
    def pull_item(self, index):
        return

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if not self.convert_flag:
            raise RuntimeError('Please finish convertion first.')
        else:
            return self.ids[index]

def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    print("Hello!")