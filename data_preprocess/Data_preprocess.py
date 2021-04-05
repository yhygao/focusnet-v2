import os
import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt

import pdb

reference_direction = np.identity(3).flatten()
reference_spacing = np.array((1., 1., 2.5))


os.chdir('./PDDCA')

for i in os.listdir('.'):
#for i in ['0522c0576']:
    print('start')
    os.chdir(i)

    image = sitk.ReadImage('img.nrrd')
    
    image_size = image.GetSize()
    image_spacing = image.GetSpacing()
    image_origin = image.GetOrigin()
    image_direction = image.GetDirection()

    
    label = np.zeros((image_size[2], image_size[0], image_size[1]))
    label_index = {'BrainStem.nrrd': 1,
                   'Chiasm.nrrd': 2,
                   'Mandible.nrrd': 3,
                   'OpticNerve_L.nrrd': 4,
                   'OpticNerve_R.nrrd': 5,
                   'Parotid_L.nrrd': 6,
                   'Parotid_R.nrrd': 7,
                   'Submandibular_L.nrrd': 8,
                   'Submandibular_R.nrrd': 9
                    }
    # generate itk label
    for j in os.listdir('./structures'):
        print(j)
        structure = sitk.ReadImage('./structures/'+j)
        npstructure = sitk.GetArrayFromImage(structure)
        label[npstructure == 1] = label_index[j]
    
    print(np.unique(label))
    
    itkLabel = sitk.GetImageFromArray(label)
    itkLabel.SetSpacing(image_spacing)
    itkLabel.SetOrigin(image_origin)
    itkLabel.SetDirection(image_direction)
        
    sitk.WriteImage(itkLabel, 'inter_label.nii.gz')
    
    new_size = np.array(image_size) * np.array(image_spacing) / reference_spacing
    reference_size = [int(new_size[0]), int(new_size[1]), int(new_size[2])]
    
    reference_image = sitk.Image(reference_size, sitk.sitkFloat32)
    reference_image.SetOrigin(image_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # set interpolator for CT data
    resampler.SetInterpolator(sitk.sitkLinear)

    out_img = resampler.Execute(image)
    # set interpolator for label
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    out_label = resampler.Execute(itkLabel)

    sitk.WriteImage(out_img, 'data.nii.gz')
    sitk.WriteImage(out_label, 'label.nii.gz')
    
    print('done')

    os.chdir('..')
