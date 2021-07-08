import torch
import torch.nn as nn
import torch.nn.functional as F

import SimpleITK as sitk
import numpy as np
import time
from skimage import measure, morphology
from utils import *
import os
import pdb 



SPACING = (1., 1., 2.5)
CLASSES = 10
SMOOTH = False
SLICE = 40
HS = SLICE // 2 # half slice

H_size = 96

def evaluation(model, itkCT, SMALL=True):

    tensorCT, center_index, origin_shape = preprocess(itkCT)
    tensorPred, heatmapPred, smallPred = predict(model, tensorCT, SMALL=SMALL)
    itkPred, itksmallPred, location_list = post_process(tensorPred, heatmapPred, smallPred, itkCT, center_index, origin_shape, SMALL=SMALL)
    
    #del tensorPred
    torch.cuda.empty_cache()
    
    return itkPred, itksmallPred, location_list#, heatmapPred

def post_process(tensorPred, heatmapPred, smallPred, itkCT, center_index, origin_shape, SMALL):
    _, _, D, H, W = origin_shape
    tensorOutput = torch.zeros(CLASSES, D, H, W).cuda()
    heatmapOutput = torch.zeros((3, D, H, W)).cuda()

    x, y = center_index
    
    tensorOutput[:, :, x-H_size:x+H_size, y-H_size:y+H_size] = tensorPred
    heatmapOutput[:, :, x-H_size:x+H_size, y-H_size:y+H_size] = heatmapPred


    tensorOutput = TorchResampleProbBySize(tensorOutput, target_size=itkCT.GetSize(), interp='trilinear')
    _, labelPred = torch.max(tensorOutput, dim=0)
    npPred = labelPred.to(torch.uint8).cpu().numpy()

    heatmapOutput = TorchResampleProbBySize(heatmapOutput, target_size=itkCT.GetSize(), interp='trilinear')
    if SMALL:
        smallOutput = torch.zeros((3, D, H, W)).cuda()
        smallOutput[:, :, x-H_size:x+H_size, y-H_size:y+H_size] = smallPred
        smallOutput = TorchResampleProbBySize(smallOutput, target_size=itkCT.GetSize(), interp='trilinear')

        smallOutput = (smallOutput>0.5).to(torch.uint8).cpu().numpy()
        npSmall = np.zeros((smallOutput.shape[1:]), dtype=np.uint8)
        npSmall[smallOutput[0, :, :, :] == 1] = 2
        npSmall[smallOutput[1, :, :, :] == 1] = 4
        npSmall[smallOutput[2, :, :, :] == 1] = 5

        itksmallPred = sitk.GetImageFromArray(npSmall)
        itksmallPred.CopyInformation(itkCT)
    else:
        itksmallPred = None

    itkPred = sitk.GetImageFromArray(npPred)
    itkPred.CopyInformation(itkCT)

    location_list = find_center_in_heatmap(heatmapOutput)

    return itkPred, itksmallPred, location_list
    
def predict(model, tensorCT, SMALL):
    _, _, D, H, W = tensorCT.shape
    tensorOutput = torch.zeros((CLASSES, D, H, W)).cuda()
    heatmapOutput = torch.zeros((3, D, H, W)).cuda()
    
    counter = torch.zeros((D, H, W)).cuda()
    one_count = torch.ones((SLICE, H, W)).cuda()

    if SMALL:
        smallOutput = torch.zeros(3, D, H, W).cuda()
        small_counter = torch.zeros(3, D, H, W).cuda()
        small_one_count = torch.ones((8, 64, 64)).cuda()
    
    with torch.no_grad():
        for i in range((D-HS)//HS):
            tensor_input = tensorCT[:, :, HS*i:HS*i+SLICE, :, :]
            tensor_input = tensor_input.cuda()

            results = model(tensor_input)

            outputs = F.softmax(results['main_result'], dim=1)
            tensorOutput[:, HS*i:HS*i+SLICE, :, :] += outputs[0, :, :, :, :]
            heatmapOutput[:, HS*i:HS*i+SLICE, :, :] += results['heatmap'].squeeze(0)
            counter[HS*i:HS*i+SLICE, :, :] += one_count

            if SMALL:
                for j in range(3):
                    if results['heatmap'][0, j, :, :, :].max() > 0.6:
                        z, x, y = results['location'][j]
                        smallOutput[j, HS*i+z-4:HS*i+z+4, x-32:x+32, y-32:y+32] += F.sigmoid(results['small_result'][0, j, :, :, :])
    
                        small_counter[j, HS*i+z-4:HS*i+z+4, x-32:x+32, y-32:y+32] += small_one_count


        tensor_input = tensorCT[:, :, -SLICE:, :, :]
        
        results = model(tensor_input)

        outputs = F.softmax(results['main_result'], dim=1)
        tensorOutput[:, -SLICE:, :, :] += outputs[0, :, :, :, :]
        heatmapOutput[:, -SLICE:, :, :] += results['heatmap'].squeeze(0)
        counter[-SLICE:, :, :] += one_count

        if SMALL:
            for j in range(3):
                if results['heatmap'][0, j, :, :, :].max() > 0.6:
                    z, x, y = results['location'][j]
                    smallOutput[j, -SLICE+z-4:-SLICE+z+4, x-32:x+32, y-32:y+32] += F.sigmoid(results['small_result'][0, j, :, :, :])
                    small_counter[j, -SLICE+z-4:-SLICE+z+4, x-32:x+32, y-32:y+32] += small_one_count
        tensorOutput = tensorOutput / counter
        heatmapOutput = heatmapOutput / counter

        if SMALL:
            smallOutput = smallOutput / small_counter
        else:
            smallOutput = None


    return tensorOutput, heatmapOutput, smallOutput


def preprocess(itkCT):
    origin_spacing = itkCT.GetSpacing()

    npImg = sitk.GetArrayFromImage(itkCT)
    npImg = (npImg + 100) / 1000.

    tensor_img = torch.from_numpy(npImg).cuda().float()
    tensor_img = TorchResampleSpacing(tensor_img, origin_spacing, target_spacing=SPACING, interp='trilinear')
    origin_shape = tensor_img.shape
    tensor_img, center_index = center_crop(tensor_img)

    return tensor_img, center_index, origin_shape


def center_crop(tensor_img):

    _, _, d, h, w = tensor_img.shape
    center_h = h // 2 #- 10
    center_w = w // 2

    tensor_img = tensor_img[:, :, :, center_h-H_size:center_h+H_size, center_w-H_size:center_w+H_size]

    return tensor_img, (center_h, center_w)

def TorchResampleSpacing(tensorCT, origin_spacing, target_spacing, interp='bilinear'):
    # resample based on spacing, interp: nearest, bilinear, trilinear
    # target spacing should follw ITK spacing order

    tensorCT = tensorCT.unsqueeze(0).unsqueeze(0)
    scale_factor = [origin_spacing[2]/target_spacing[2], origin_spacing[0]/target_spacing[0], origin_spacing[1]/target_spacing[1]]


    tensorCT = F.interpolate(tensorCT, scale_factor=scale_factor, mode=interp)
    tensorCT = tensorCT

    return tensorCT

def TorchResampleProbBySize(tensorCT, target_size, interp='trilinear'):
    # tensorCT: CLASSES, D, H, W
    size = (target_size[2], target_size[0], target_size[1])

    tensorCT = tensorCT.unsqueeze(0)
    tensorCT = F.interpolate(tensorCT, size, mode=interp)

    if SMOOTH:
        filters = torch.ones(CLASSES, 1, 3, 3, 3).cuda() / 27
        tensorCT = F.conv3d(tensorCT, filters, padding=1, groups=CLASSES)
    tensorCT = tensorCT.squeeze(0)

    return tensorCT

