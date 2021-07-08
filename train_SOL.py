import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.model import focus_net
from dataset.focusnet_dataset import BrainDataset

from torch.utils import data
from losses import FocalLoss, DiceLoss
from focusnet_validation import evaluation
from utils import *
from optparse import OptionParser
import SimpleITK as sitk

from torch.utils.tensorboard import SummaryWriter
import time
import os
import pdb


def train_net(net, options):

    data_path = options.data_path + '240dataset/'
    csv_file = options.data_path + 'new_train.csv'
    origin_spacing_data_path = options.data_path + 'origin_spacing_croped/'

    
    # z_size is the random crop size along z-axis, you can set it larger if have enough gpu memory
    trainset = BrainDataset(csv_file, data_path, data_path, mode='train', z_size=40, sigma=5, heatmap_on=True, focus_on_small=False)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=0)

    test_data_list, test_label_list, test_center_list = load_test_data(origin_spacing_data_path)

    writer = SummaryWriter(options.log_path + options.unique_name)
    
    main_params = []
    SOL_params = []

    for child, module in net.named_children():
        if child == 'SOL':
            SOL_params += list(module.parameters())
        else:
            main_params += list(module.parameters())



    optimizer_SOL = optim.Adam(SOL_params, lr=0.0004, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_SOL, milestones=[100, 200], gamma=0.25)
    criterion_mse = nn.MSELoss()


    best_dis = 1000
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0
        epoch_heatloss = 0

        net.train()
        for i, (img, label, weight, heatmap) in enumerate(trainLoader, 0):

            img = img.cuda()
            label = label.cuda()
            weight = weight.cuda()
            heatmap = heatmap.cuda()

            end = time.time()

            optimizer_SOL.zero_grad()

            result = net(img)

            loss_heatmap = criterion_mse(result['heatmap'], heatmap)
            
            loss_heatmap.backward()

            optimizer_SOL.step()


            epoch_heatloss += loss_heatmap.item()

            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss_heatmap.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Heat_loss', epoch_heatloss/(i+1), epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if (epoch+1)%10==0:
            torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
            
        dice_list, distance_list = validation(net, test_data_list, test_label_list, test_center_list)
        avg_dis = distance_list.mean()
        
        scheduler.step()
        writer.add_scalar('Test_distance/AVG_dis', distance_list.mean(), epoch+1)
        for idx in range(3):
            writer.add_scalar('Test_distance/dis%d'%(idx+1), distance_list[idx], epoch+1)


        if avg_dis <= best_dis:
            best_dis = avg_dis
            torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

        print('save done')
        print('dis: %.5f/best dis: %.5f'%(avg_dis, best_dis))


def load_test_data(data_path):
    test_name_list = ['0522c0555', '0522c0576', '0522c0598', '0522c0659', '0522c0661',
                    '0522c0667', '0522c0669', '0522c0708', '0522c0727', '0522c0746']
                    #'0522c0788', '0522c0806', '0522c0845', '0522c0857', '0522c0878']
    #test_name_list = ['0522c0857']
    test_data_list = []
    test_label_list = []
    test_center_list = []


    for name in test_name_list:
        CT = sitk.ReadImage(data_path + name + '_data.nii.gz')
        label = sitk.ReadImage(data_path + name + '_label.nii.gz')

        center = find_center(label)
        test_data_list.append(CT)
        test_label_list.append(label)
        test_center_list.append(center)

    return test_data_list, test_label_list, test_center_list


def validation(net, test_data_list, test_label_list, test_center_list):
    
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()

    total_dice = 0 
    dice_list = np.zeros(9)
    distance_list = np.zeros(3)
    small_index = [2, 4, 5]

    for i in range(len(test_data_list)):
        tmp_dice_list = np.zeros(9)

        itkCT = test_data_list[i]
        gt_center = test_center_list[i]

        itkLabel = test_label_list[i]
        itkPred, itksmallPred, pred_center = evaluation(net, itkCT, SMALL=False)
    
        tmp_distance_list = cal_distance(pred_center, gt_center, itkCT)
    
        for idx in range(1, 10):
            dicecomputer.Execute(itkLabel==idx, itkPred==idx)
            dice = dicecomputer.GetDiceCoefficient()
            tmp_dice_list[idx-1] += dice


        print('dice', tmp_dice_list.mean(), 'distance', tmp_distance_list.mean())

        dice_list += tmp_dice_list
        distance_list += tmp_distance_list

    dice_list /= len(test_data_list)
    distance_list /= len(test_data_list)
    
    print('avg dice:', dice_list.mean())
    return dice_list, distance_list




if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=300, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=1,
            type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default='./checkpoint/s-net/CP359.pth',
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path',
            default='./log/', help='log path')
    parser.add_option('--data_path', type='str', dest='data_path',
            default='/research/cbim/vast/yg397/OAR/dataset/MICCAI2015_dataset/', help='data_path')
    parser.add_option('-m', type='str', dest='model',
            default='focus_net', help='use which model')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='test', help='use which model')
    parser.add_option('--rlt', type='float', dest='rlt',
            default=0.2, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='org_weight',
            default=[0.5,1,8,1,8,8,1,1,2,2], help='weight of focal loss')
    parser.add_option('--norm', type='str', dest='norm',
            default='bn')
    parser.add_option('--gpu', type='str', dest='gpu',
            default='0')
    
    (options, args) = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    print('use model:', options.model)
    if options.model == 'focus_net':
        net = focus_net(1, 10, se=True, norm=options.norm, SOSNet=False)
    else:
        print('wrong model')

    if options.load:
        pth = torch.load(options.load)
        net = load_my_state_dict(net, pth)
        print('Model loaded from {}'.format(options.load))
    net.cuda()
    train_net(net, options)

    print('done')

