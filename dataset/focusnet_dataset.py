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
    def __init__(self, csv_file, dataset_dir, label_dir, mode='train', sigma=5, z_size=40, heatmap_on=False, focus_on_small=False):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        colnames = ['data', 'weight']
        self.csv = pd.read_csv(csv_file, header=None, names=colnames)
        self.data_index = self.csv.data.tolist()
        self.weightList = self.csv.weight.tolist()
        self.sigma = sigma
        self.z_slice = z_size
        self.heatmap_on = heatmap_on
        self.focus_on_small = focus_on_small

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
        if self.heatmap_on:
            print('start generate heatmap')
            self.heatmapList = self._generate_heatmap(self.labelList)
            print('generate done')

        
    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        #with torch.cuda.device(1):
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
        if self.heatmap_on:
            tensor_heatmap = self.heatmapList[idx].cuda().float().unsqueeze(0)
        else:
            tensor_heatmap = None


        if self.mode == 'train':
            tensor_image, tensor_label, tensor_heatmap = self.random_zoom_rotate(tensor_image, tensor_label, tensor_heatmap)

        tensor_image, tensor_label, tensor_heatmap = self.randcrop(tensor_image, tensor_label, tensor_heatmap)
        assert tensor_image.shape == tensor_label.shape

        if self.heatmap_on:
            return tensor_image.float(), tensor_label.long(), tensor_weight.float(), tensor_heatmap.float()
        else:
            return tensor_image.float(), tensor_label.long(), tensor_weight.float()



    def randcrop(self, img, label, heatmap):
        _, _, D, H, W = img.shape
        
        diff_D = D - self.z_slice
        diff_H = H - 192
        diff_W = W - 192
        
        if self.mode == 'train':
            if self.focus_on_small:
                if np.random.random() > 0.1: 
                    rand_z = np.random.randint(diff_D-10, diff_D)
                else:
                    rand_z = np.random.randint(0, diff_D)
            else:
                rand_z = np.random.randint(0, diff_D)
            rand_x = np.random.randint(0, diff_H)
            rand_y = np.random.randint(0, diff_W)
            
            croped_img = img[0, :, rand_z:rand_z+self.z_slice, rand_x:rand_x+192, rand_y:rand_y+192]
            croped_lab = label[0, :, rand_z:rand_z+self.z_slice, rand_x:rand_x+192, rand_y:rand_y+192]
            if heatmap is not None:
                croped_heatmap = heatmap[0, :, rand_z:rand_z+self.z_slice, rand_x:rand_x+192, rand_y:rand_y+192]

        else:
            if self.focus_on_small:
                if np.random.random() > 0.1:
                    rand_z = np.random.randint(diff_D-10, diff_D)
                else:
                    rand_z = np.random.randint(0, diff_D)
            else:
                rand_z = np.random.randint(0, diff_D)

            rand_x = np.random.randint(0, diff_H)
            rand_y = np.random.randint(0, diff_W)

            croped_img = img[0, :, rand_z:rand_z+self.z_slice, rand_x:rand_x+192, rand_y:rand_y+192]
            croped_lab = label[0, :, rand_z:rand_z+self.z_slice, rand_x:rand_x+192, rand_y:rand_y+192]
            if heatmap is not None:
                croped_heatmap = heatmap[0, :, rand_z:rand_z+self.z_slice, rand_x:rand_x+192, rand_y:rand_y+192]
        if heatmap is None:
            croped_heatmap = None

        return croped_img, croped_lab, croped_heatmap


    def random_zoom_rotate(self, img, label, heatmap):
        scale_z = np.random.random() * 0.2 + 0.9
        scale_x = np.random.random() * 0.2 + 0.9    # 0.6 0.7
        scale_y = np.random.random() * 0.2 + 0.9


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
        grid = F.affine_grid(theta, img.size()).cuda()
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border')
        label = F.grid_sample(label.float(), grid, mode='nearest').long()
        if heatmap is not None:
            heatmap = F.grid_sample(heatmap, grid, mode='bilinear').float()
    
        return img, label, heatmap

    
    def init_pad(self, npImg, npLab):
        d, h, w = npImg.shape

        if d < 75:
            diff = 75 - d
            npImg = np.pad(npImg, ((diff, 0), (0, 0), (0,0)), mode='edge')
            npLab = np.pad(npLab, ((diff, 0), (0, 0), (0, 0)), mode='constant')

        return npImg, npLab

    def _generate_heatmap(self, labelList):

        center_map_list = []
        for label in labelList:
            D, H, W = label.shape
            center_map = np.zeros([3, D, H, W])
            region = regionprops(label)
            for i in range(len(region)):
                if region[i].label == 2:
                    (z, x, y) = region[i].centroid
                    z, x, y = int(round(z)), int(round(x)), int(round(y))
                    center_map[0, z, x, y] = 1
                if region[i].label == 4:
                    (z, x, y) = region[i].centroid
                    z, x, y = int(round(z)), int(round(x)), int(round(y))
                    center_map[1, z, x, y] = 1
                if region[i].label == 5:
                    (z, x, y) = region[i].centroid
                    z, x, y = int(round(z)), int(round(x)), int(round(y))
                    center_map[2, z, x, y] = 1
            center_map_list.append(center_map)

        heatmap_list = self._gaussian_map(center_map_list)

        return heatmap_list

    def _gaussian_map(self, center_map_list):
        heatmap_list = []

        for center_map in center_map_list:
            _, d, h, w = center_map.shape
            D1 = torch.linspace(1, d, d)
            H1 = torch.linspace(1, h, h)
            W1 = torch.linspace(1, w, w)
            [D, H, W] = torch.meshgrid(D1, H1, W1)
            z, x, y = np.where(center_map[0, :, :, :])
            DD = D - (z[0]+1)
            HH = H - x[0]
            WW = W - y[0]

            cube = DD*DD + HH*HH + WW*WW
            cube /= (2. * self.sigma * self.sigma)
            cube = torch.exp(-cube)
            cube[0:z[0]-1, :, :] = 0
            cube[z[0]+2:, :, :] = 0
            cube = cube.unsqueeze(0)

            for i in range(1, 3):
                z, x, y = np.where(center_map[i, :, :, :])
                DD = D - (z[0]+1)
                HH = H - x[0]
                WW = W - y[0]

                tmp_cube = DD*DD + HH*HH + WW*WW
                tmp_cube /= ( 2. * self.sigma * self.sigma)
                tmp_cube = torch.exp(-tmp_cube)
                tmp_cube[0:z[0]-1, :, :] = 0
                tmp_cube[z[0]+2:, :, :] = 0
                tmp_cube = tmp_cube.unsqueeze(0)
                cube = torch.cat((cube, tmp_cube), dim=0)

            cube[cube < 0.01] = 0
            heatmap_list.append(cube)
        return heatmap_list
            
    
    
    


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_path = '/data/head/MICCAI2015/dataset/240dataset/'
    csv_file = '/data/head/MICCAI2015/dataset/new_test.csv'

    dataset = BrainDataset(csv_file, data_path, data_path, mode='train', heatmap_on=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (img, label, weight, heatmap) in enumerate(dataloader):
        print(i, img.shape, label.shape, heatmap.shape, weight )
        if heatmap.max() == 0:
            continue
        for j in range(img.shape[2]):
            if heatmap[0, 0, j, :, :].max() < 0.1:
                continue
            else:
                print(heatmap[0, 0, j, :, :].max())
            plt.subplot(2,3,1)
            plt.imshow(img.cpu().numpy()[0, 0, j, :, :])
            plt.subplot(2,3,2)
            plt.imshow(label.cpu().numpy()[0, 0, j, :, :])
            plt.subplot(2,3,3)
            plt.imshow(heatmap.cpu().numpy()[0, 0, j, :, :])
            plt.subplot(2,3,4)
            plt.imshow(heatmap.cpu().numpy()[0, 1, j, :, :])
            plt.subplot(2,3,5)
            plt.imshow(heatmap.cpu().numpy()[0, 2, j, :, :])

            plt.subplot(2,3,6)
            plt.imshow(heatmap.cpu().numpy()[0, 0, j, :, :] + heatmap.cpu().numpy()[0, 1, j, :, :] + heatmap.cpu().numpy()[0, 2, j, :, :])




            plt.show()
