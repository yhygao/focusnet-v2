import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.model import s_net
from dataset.brain_dataset import BrainDataset

from torch.utils import data
from losses import FocalLoss, DiceLoss
from validation import evaluation
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
    trainset = BrainDataset(csv_file, data_path, data_path, mode='train', z_size=40)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=0)

    test_data_list, test_label_list = load_test_data(origin_spacing_data_path)

    writer = SummaryWriter(options.log_path + options.unique_name)
    
    
    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.0005)

    org_weight = torch.FloatTensor(options.org_weight).unsqueeze(1).cuda()
    criterion_fl = FocalLoss(10, alpha=org_weight)
    criterion_dl = DiceLoss()

    best_dice = 0
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        multistep_scheduler = multistep_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, lr_decay_epoch=[200, 400], max_epoch=options.epochs, gamma=0.1)
        print('current lr:', multistep_scheduler)
        
        net.train()
        for i, (img, label, weight) in enumerate(trainLoader, 0):

            img = img.cuda()
            label = label.cuda()
            weight = weight.cuda()

            end = time.time()

            optimizer.zero_grad()

            result = net(img)

            if options.rlt > 0:
                loss = criterion_fl(result, label, weight) + options.rlt * criterion_dl(result, label, weight)
            else:
                loss = criterion_dl(result, label, weight)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', multistep_scheduler, epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if (epoch+1)%10==0:
            torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
            
        avg_dice, dice_list = validation(net, test_data_list, test_label_list)
        writer.add_scalar('Test/AVG_Dice', avg_dice, epoch+1)
        for idx in range(9):
            writer.add_scalar('Test/Dice%d'%(idx+1), dice_list[idx], epoch+1)

        if avg_dice >= best_dice:
            best_dice = avg_dice
            torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

        print('save done')
        print('dice: %.5f/best dice: %.5f'%(avg_dice, best_dice))

def load_test_data(data_path):
    test_name_list = ['0522c0555', '0522c0576', '0522c0598', '0522c0659', '0522c0661',
                    '0522c0667', '0522c0669', '0522c0708', '0522c0727', '0522c0746',
                    ]
    test_data_list = []
    test_label_list = []

    for name in test_name_list:
        CT = sitk.ReadImage(data_path + name + '_data.nii.gz')
        label = sitk.ReadImage(data_path + name + '_label.nii.gz')

        test_data_list.append(CT)
        test_label_list.append(label)

    return test_data_list, test_label_list
    

def validation(net, test_data_list, test_label_list):
    
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()

    total_dice = 0
    dice_list = np.zeros(9)
    for i in range(len(test_data_list)):
        tmp_dice_list = np.zeros(9)
        itkCT = test_data_list[i]
        itkLabel = test_label_list[i]

        itkPred = evaluation(net, itkCT)
        for idx in range(1, 10):
            dicecomputer.Execute(itkLabel==idx, itkPred==idx)
            dice = dicecomputer.GetDiceCoefficient()
            tmp_dice_list[idx-1] += dice

        print('dice', tmp_dice_list.mean())
        dice_list += tmp_dice_list

    dice_list /= len(test_data_list)

    return dice_list.mean(), dice_list


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=600, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=1,
            type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path',
            default='./log/', help='log path')
    parser.add_option('--data_path', type='str', dest='data_path',
            default='/research/cbim/vast/yg397/OAR/dataset/MICCAI2015_dataset/', help='data_path')
    parser.add_option('-m', type='str', dest='model',
            default='s_net', help='use which model')
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
    if options.model == 's_net':
        net = s_net(1, 10, se=True, norm=options.norm)
    else:
        print('wrong model')

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    net.cuda()
    train_net(net, options)

    print('done')

