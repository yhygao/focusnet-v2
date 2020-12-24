import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from skimage.measure import regionprops

import pdb 

def load_my_state_dict(net, pretrained_dict):
    
    model_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    return net 


def find_center(itkLabel):
    npLabel = sitk.GetArrayFromImage(itkLabel)

    centerList = []
    region = regionprops(npLabel)
    for i in range(len(region)):
        if region[i].label == 2:
            centerList.append(region[i].centroid)
        elif region[i].label == 4:
            centerList.append(region[i].centroid)
        elif region[i].label == 5:
            centerList.append(region[i].centroid)
        else:
            pass

    return centerList

def find_center_in_heatmap(tensor_heatmap):
    c, d, h, w  = tensor_heatmap.shape
    location_list = []
    for i in range(c):
        index = torch.argmax(tensor_heatmap[i, :, :, :])

        z = int(index // w // h)
        index -= z * w * h

        x = int(index // h)
        index -= x * h

        y = int(index)

        location_list.append((z, x, y))

    return location_list



def cal_distance(pred_center, gt_center, itkCT):
    space_x, space_y, space_z = itkCT.GetSpacing()
    distance = np.zeros(3)
    for i in range(3):
        tmp_pred_center = pred_center[i]
        tmp_gt_center = gt_center[i]
        
        z_sp = (tmp_pred_center[0] - tmp_gt_center[0]) * space_z
        y_sp = (tmp_pred_center[1] - tmp_gt_center[1]) * space_y
        x_sp = (tmp_pred_center[2] - tmp_gt_center[2]) * space_x

        tmp_distance = np.sqrt(z_sp**2 + y_sp**2 + x_sp**2)
        distance[i] = tmp_distance

    return distance


def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def cal_dice(pred, target, C): 
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.cuda()
    dice = 2 * intersection / summ

    return dice, intersection, summ

