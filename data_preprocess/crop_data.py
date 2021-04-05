import numpy as np
import SimpleITK as sitk
import os
import csv

from skimage import measure, color
import matplotlib.pyplot as plt

import pdb


def crop(image, center_stem, center_parotid, center_nerve):
    
    z, x, y = image.shape
    z_bottom = int(center_parotid[0]) - 37
    z_upper = int(center_nerve[0]) + 12

    center_x, center_y = int(center_stem[1]), int(center_stem[2])

    crop_data = image[z_bottom:z_upper, center_x-120-50:center_x+120-50, center_y-120:center_y+120]

    return crop_data

os.chdir('./PDDCA')

for i in os.listdir('.'):

    os.chdir(i)
    print(i)

    data = sitk.ReadImage('data.nii.gz')
    label = sitk.ReadImage('label.nii.gz')

    origin = data.GetOrigin()
    spacing = data.GetSpacing()
    
    assert data.GetSpacing() == (1.0, 1.0, 2.5)
    assert label.GetSpacing() == (1.0, 1.0, 2.5)

    npdata = sitk.GetArrayFromImage(data)
    nplabel = sitk.GetArrayFromImage(label).astype(np.uint8)

    D, H, W = nplabel.shape

    label_stem = nplabel * (nplabel==1)
    region = measure.regionprops(label_stem)

    print('len of region:', len(region))
    center_stem = region[0].centroid
    print(center_stem)
    
    label_parotid = nplabel * (nplabel==7)
    region = measure.regionprops(label_parotid)
    
    center_parotid = region[0].centroid
    print(center_parotid)

    label_nerve = nplabel * (nplabel==4)
    region = measure.regionprops(label_nerve)

    center_nerve = region[0].centroid
    print(center_nerve)

    crop_data = crop(npdata, center_stem, center_parotid, center_nerve)
    crop_label = crop(nplabel, center_stem, center_parotid, center_nerve)

    itkdata = sitk.GetImageFromArray(crop_data)
    itklabel = sitk.GetImageFromArray(crop_label)

    itkdata.SetOrigin(origin)
    itkdata.SetSpacing(spacing)

    itklabel.SetOrigin(origin)
    itklabel.SetSpacing(spacing)

    sitk.WriteImage(itkdata, '/data/head/MICCAI2015/dataset/240dataset/%s_data.nii.gz'%i)
    sitk.WriteImage(itklabel, '/data/head/MICCAI2015/dataset/240dataset/%s_label.nii.gz'%i)

    print('done')
    
    os.chdir('..')
