import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops
from scipy import signal
import math
import re
import pdb

class BrainDataset(Dataset):
    def __init__(self, csv_file, dataset_dir, label_dir, mode='train', sigma=8, z_size=30):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        colnames = ['data', 'weight']
        self.csv = pd.read_csv(csv_file, header=None, names=colnames)
        self.data_index = self.csv.data.tolist()
        self.weightList = self.csv.weight.tolist()
        self.sigma = sigma
        self.z_size = z_size

        self.dataList = []
        self.labelList = []

        print('start loading data')
        for i in range(len(self.data_index)):
            img = sitk.ReadImage(self.dataset_dir + self.data_index[i] + '_data.nii.gz')
            npImg = sitk.GetArrayFromImage(img)
            npImg = (npImg+100) / 1000

            lab = sitk.ReadImage(self.label_dir + self.data_index[i] + '_label.nii.gz')
            npLab = sitk.GetArrayFromImage(lab)

            npImg, npLab = self.init_pad(npImg, npLab)        

            self.dataList.append(npImg)
            self.labelList.append(npLab)

        print('load done, length of dataset:', len(self.dataList))
        
    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        image = self.dataList[idx]
        label = self.labelList[idx]
        tmp = self.weightList[idx]
        tmp = re.split('\.|\[|\]', tmp)
        while '' in tmp:
            tmp.remove('')
        weight = np.array(tmp).astype(np.float32)
        
        tensor_image = torch.from_numpy(image).cuda().float().unsqueeze(0).unsqueeze(0)
        tensor_label = torch.from_numpy(label).cuda().long().unsqueeze(0).unsqueeze(0)
        tensor_weight = torch.from_numpy(weight).cuda().float()
        
        if self.mode == 'train':
            tensor_image, tensor_label = self.random_zoom_rotate(tensor_image, tensor_label)

        tensor_image, tensor_label = self.randcrop(tensor_image, tensor_label)
        assert tensor_image.shape == tensor_label.shape

        return tensor_image, tensor_label, tensor_weight

    def randcrop(self, img, label):
        _, _, D, H, W = img.shape
        
        diff_D = D - self.z_size
        diff_H = H - 192
        diff_W = W - 192
        
        if self.mode == 'train':
            rand_z = np.random.randint(0, diff_D)
            rand_x = np.random.randint(0, diff_H)
            rand_y = np.random.randint(0, diff_W)
            
            croped_img = img[0, :, rand_z:rand_z+self.z_size, rand_x:rand_x+192, rand_y:rand_y+192]
            croped_lab = label[0, :, rand_z:rand_z+self.z_size, rand_x:rand_x+192, rand_y:rand_y+192]

        else:
            rand_x = diff_H // 2
            rand_y = diff_W // 2

            croped_img = img[0, :, :, rand_x:rand_x+192, rand_y:rand_y+192]
            croped_lab = label[0, :, :, rand_x:rand_x+192, rand_y:rand_y+192]


        return croped_img, croped_lab


    def random_zoom_rotate(self, img, label):
        scale_z = np.random.random() * 0.2 + 0.9
        scale_x = np.random.random() * 0.4 + 0.8
        scale_y = np.random.random() * 0.4 + 0.8


        theta_scale = torch.tensor([[scale_z, 0, 0, 0],
                                    [0, scale_x, 0, 0],
                                    [0, 0, scale_y, 0],
                                    [0, 0, 0, 1]]).float().cuda()
        angle_z = (float(np.random.randint(-5, 5)) / 180.) * math.pi
        angle_x = (float(np.random.randint(-10, 10)) / 180.) * math.pi
        angle_y = (float(np.random.randint(-10, 10)) / 180.) * math.pi

        theta_rotate_z = torch.tensor( [[1, 0, 0, 0], 
                                        [0, math.cos(angle_z), -math.sin(angle_z), 0], 
                                        [0, math.sin(angle_z), math.cos(angle_z), 0], 
                                        [0, 0, 0, 1]]).float().cuda()
        theta_rotate_x = torch.tensor( [[math.cos(angle_x), 0, math.sin(angle_x), 0], 
                                        [0, 1, 0, 0], 
                                        [-math.sin(angle_x), 0, math.cos(angle_x), 0], 
                                        [0, 0, 0, 1]]).float().cuda()
        theta_rotate_y = torch.tensor( [[math.cos(angle_y), -math.sin(angle_y), 0, 0], 
                                        [math.sin(angle_y), math.cos(angle_y), 0, 0], 
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]]).float().cuda()
        theta = torch.mm(theta_rotate_z, theta_rotate_x)
        theta = torch.mm(theta, theta_rotate_y)
        theta = torch.mm(theta, theta_scale)[0:3, :]
    
        theta = theta.unsqueeze(0)
        grid = F.affine_grid(theta, img.size(), align_corners=True).cuda()
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        label = F.grid_sample(label.float(), grid, mode='nearest', align_corners=True).long()
    
        return img, label





   
    def init_pad(self, npImg, npLab):
        d, h, w = npImg.shape

        if d < 75:
            diff = 75 - d
            npImg = np.pad(npImg, ((diff, 0), (0, 0), (0,0)), mode='edge')
            npLab = np.pad(npLab, ((diff, 0), (0, 0), (0, 0)), mode='constant')

        return npImg, npLab

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = '/data/head/MICCAI2015/dataset/240dataset/'
    csv_file = '/data/head/MICCAI2015/dataset/new_test.csv'

    dataset = BrainDataset(csv_file, data_path, data_path, mode='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (img, label, weight) in enumerate(dataloader):
        print(i, img.shape, label.shape, weight)
