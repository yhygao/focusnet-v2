import torch
import torch.nn as nn
import torch.nn.functional as F

import SimpleITK as sitk
import numpy as np
import time
from skimage import measure, morphology

import os
import pdb


SPACING = (1., 1., 2.5)
CLASSES = 10
SMOOTH = False
SLICE = 40
HS = SLICE // 2 # half slice

def evaluation(model, itkCT):

    tensorCT, center_index, origin_shape = preprocess(itkCT)
    tensorPred = predict(model, tensorCT)
    itkPred = post_process(tensorPred, itkCT, center_index, origin_shape)
    
    del tensorPred
    torch.cuda.empty_cache()
    return itkPred

def post_process(tensorPred, itkCT, center_index, origin_shape):
    _, _, D, H, W = origin_shape
    tensorOutput = torch.zeros(CLASSES, D, H, W).cuda()

    x, y = center_index

    tensorOutput[:, :, x-96:x+96, y-96:y+96] = tensorPred

    tensorOutput = TorchResampleProbBySize(tensorOutput, target_size=itkCT.GetSize(), interp='trilinear')
    _, labelPred = torch.max(tensorOutput, dim=0)
    npPred = labelPred.to(torch.uint8).cpu().numpy()

    itkPred = sitk.GetImageFromArray(npPred)
    itkPred.CopyInformation(itkCT)

    return itkPred
    
def predict(model, tensorCT):
    _, _, D, H, W = tensorCT.shape
    tensorOutput = torch.zeros(CLASSES, D, H, W).cuda()
    with torch.no_grad():
        for i in range((D-HS)//HS):
            tensor_input = tensorCT[:, :, HS*i:HS*i+SLICE, :, :]
            outputs = model(tensor_input)
            outputs = F.softmax(outputs, dim=1)
            tensorOutput[:, HS*i:HS*i+SLICE, :, :] += outputs[0, :, :, :, :]
        
        tensor_input = tensorCT[:, :, -SLICE:, :, :]
        outputs = model(tensor_input)
        outputs = F.softmax(outputs, dim=1)
        tensorOutput[:, -SLICE:, :, :] += outputs[0, :, :, :, :]



    return tensorOutput


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
    center_h = h // 2
    center_w = w // 2

    tensor_img = tensor_img[:, :, :, center_h-96:center_h+96, center_h-96:center_h+96]

    return tensor_img, (center_h, center_w)

def TorchResampleSpacing(tensorCT, origin_spacing, target_spacing, interp='bilinear'):
    # resample based on spacing, interp: nearest, bilinear, trilinear
    # target spacing should follw ITK spacing order

    tensorCT = tensorCT.unsqueeze(0).unsqueeze(0)
    scale_factor = [origin_spacing[2]/target_spacing[2], origin_spacing[0]/target_spacing[0], origin_spacing[1]/target_spacing[1]]


    tensorCT = F.interpolate(tensorCT, scale_factor=scale_factor, mode=interp, align_corners=True)
    tensorCT = tensorCT

    return tensorCT

def TorchResampleProbBySize(tensorCT, target_size, interp='trilinear'):
    # tensorCT: CLASSES, D, H, W
    size = (target_size[2], target_size[0], target_size[1])

    tensorCT = tensorCT.unsqueeze(0)
    tensorCT = F.interpolate(tensorCT, size, mode=interp, align_corners=True)

    if SMOOTH:
        filters = torch.ones(CLASSES, 1, 3, 3, 3).cuda() / 27
        tensorCT = F.conv3d(tensorCT, filters, padding=1, groups=CLASSES)
    tensorCT = tensorCT.squeeze(0)

    return tensorCT


