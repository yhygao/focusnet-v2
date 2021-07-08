import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.model import focus_net
from model.AE3D import ConvAE

from dataset.focusnet_dataset import BrainDataset

from torch.utils import data
from losses import FocalLoss, DiceLoss, BinaryDiceLoss, BinaryFocalLoss
from focusnet_validation import evaluation
from utils import *
from optparse import OptionParser
import SimpleITK as sitk

#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import time
import math
import os
import pdb 



AE_ON = True



def train_net(net, AENet, options):

    data_path = options.data_path + '240dataset/'
    csv_file = options.data_path + 'new_train.csv'
    origin_spacing_data_path = options.data_path + 'origin_spacing_croped/'

    trainset = BrainDataset(csv_file, data_path, data_path, mode='train', heatmap_on=False, focus_on_small=True)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=0)

    test_data_list, test_label_list, test_center_list = load_test_data(origin_spacing_data_path)

    writer = SummaryWriter(options.log_path + options.unique_name)
    
   
    small_params = []
    for child, module in net.named_children():
        if child == 'SOS':
            small_params += list(module.parameters())
        else:
            for param in list(module.parameters()):
                param.requires_grad = False


    for param in small_params:
        param.requires_grad = True
 
    #_, small_dice_list, _ = validation(net, test_data_list, test_label_list, test_center_list)
    #print(small_dice_list)

    optimizer_G = optim.Adam(small_params, lr=2e-4, weight_decay=0.0001)
    optimizer_D = optim.Adam(AENet.parameters(), lr=2e-4, weight_decay=0.0001)


    criterion_bdl = BinaryDiceLoss().cuda()
    criterion_bfl = BinaryFocalLoss(alpha=0.5).cuda()


    best_dice = 0
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))

        g_epoch_loss = 0
        d_epoch_loss = 0
        
        epoch_seg_loss = 0
        epoch_AE_loss = 0
        epoch_cls_loss = 0
        epoch_recon_loss = 0
        epoch_dis_loss = 0

        for i, (img, label, weight) in enumerate(trainLoader, 0):

            img = img.cuda().float()
            label = label.cuda().float()

            end = time.time()
            net.train()





            result = net(img, label=label)
            
            num = result['small_result'].shape[1]
            valid = torch.randn((num, 1), requires_grad=False).fill_(1.0).cuda()
            fake = torch.randn((num, 1), requires_grad=False).fill_(0.0).cuda()
           
            pred_small = result['small_result'].permute(1,0,2,3,4)
            true_small = result['small_label'].permute(1,0,2,3,4)


            # --------------------------------------
            # Train discriminator
            # --------------------------------------
            recon_pred, code_pred, prob_pred, feat_pred = AENet(F.sigmoid(pred_small.detach()))
            recon_true, code_true, prob_true, feat_true = AENet(true_small)

           
            optimizer_D.zero_grad()
            
            loss_recon , loss_dis = calc_discriminator_loss(pred_small.detach(), recon_pred, code_pred, prob_pred, feat_pred, true_small, recon_true, code_true, prob_true, feat_true, valid, fake)

            print('D', loss_recon.item(), loss_dis.item())
            loss_d = loss_recon + loss_dis
            loss_d.backward()
            d_epoch_loss += loss_d.item()
            epoch_recon_loss += loss_recon.item()
            epoch_dis_loss += loss_dis.item()

            optimizer_D.step()


            
            # ----------------------------------------
            # Train generator
            # ----------------------------------------
            recon_pred, code_pred, prob_pred, feat_pred = AENet(F.sigmoid(pred_small))
            recon_true, code_true, prob_true, feat_true = AENet(true_small)


           
            loss_seg, loss_AE, loss_cls = calc_seg_loss(pred_small, recon_pred, code_pred, prob_pred, feat_pred, true_small, recon_true, code_true, prob_true, feat_true, criterion_bfl, criterion_bdl, valid)
            

            print('G', loss_seg.item(), loss_AE.item(), loss_cls)
            loss_g = loss_seg + loss_AE + loss_cls
            epoch_seg_loss += loss_seg
            epoch_AE_loss += loss_AE
            epoch_cls_loss += loss_cls
            
            
            optimizer_G.zero_grad()
            loss_g.backward()
            optimizer_G.step()
            g_epoch_loss += loss_g.item()



            batch_time = time.time() - end
            print('g_batch_loss: %.5f, d_batch_loss: %.5f, batch_time:%.5f'%(loss_g.item(), loss_d.item(), batch_time))

        print('[epoch %d] g_epoch-loss: %.5f, d_epoch_loss: %.5f'%(epoch+1, g_epoch_loss/(i+1), d_epoch_loss/(i+1)))

        writer.add_scalar('Train/g_loss', g_epoch_loss/(i+1), epoch+1)
        writer.add_scalar('Train/d_loss', d_epoch_loss/(i+1), epoch+1)
        writer.add_scalar('Train/seg_loss', epoch_seg_loss/(i+1), epoch+1)
        writer.add_scalar('Train/AE_loss', epoch_AE_loss/(i+1), epoch+1)
        writer.add_scalar('Train/cls_loss', epoch_cls_loss/(i+1), epoch+1)
        writer.add_scalar('Train/recon_loss', epoch_recon_loss/(i+1), epoch+1)
        writer.add_scalar('Train/dis_loss', epoch_dis_loss/(i+1), epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if (epoch+1) % 10 == 0:

            torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch+1))
            torch.save(AENet.state_dict(), '%s%s/AE_CP%d.pth'%(options.cp_path, options.unique_name, epoch+1))
            torch.cuda.empty_cache()
            
        if (epoch+1) % 1 == 0:
            
            _, small_dice_list, _ = validation(net, test_data_list, test_label_list, test_center_list)
            avg_dice = small_dice_list.mean()
            if avg_dice >= best_dice:
                best_dice = avg_dice
                torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

            writer.add_scalar('Test_small/AVG_Dice', avg_dice, epoch+1)
            for idx in range(3):
                writer.add_scalar('Test_small/Dice%d'%(idx+1), small_dice_list[idx], epoch+1)


            print('save done')
            print('dice: %.5f/best dice: %.5f'%(avg_dice, best_dice))



def calc_discriminator_loss(pred_small, recon_pred, code_pred, prob_pred, feat_pred, true_small, recon_true, code_true, prob_true, feat_true, valid, fake):
    
    loss_recon = 0.999*(F.mse_loss(recon_pred, torch.sigmoid(pred_small.detach())) + F.mse_loss(recon_true, true_small))
    loss_dis = -0.001*(F.mse_loss(code_pred, code_true) + 0.2*F.mse_loss(feat_pred[0], feat_true[0]) + 0.2*F.mse_loss(feat_pred[1], feat_true[1]))


    return loss_recon, loss_dis 



    



def calc_seg_loss(pred_small, recon_pred, code_pred, prob_pred, feat_pred, true_small, recon_true, code_true, prob_true, feat_true, criterion_bfl, criterion_bdl, valid):

    loss_seg = criterion_bfl(pred_small, true_small)
    loss_seg += 0.5*criterion_bdl(pred_small, true_small)
    loss_AE = 5*F.mse_loss(code_pred, code_true)
    loss_AE += F.mse_loss(feat_pred[0], feat_true[0]) + F.mse_loss(feat_pred[1], feat_true[1]) + F.mse_loss(F.sigmoid(pred_small), true_small)
    return loss_seg, loss_AE, 0#loss_cls


        

    
def load_test_data(data_path):
    
    test_name_list = [
    '0522c0555', '0522c0576', '0522c0598', '0522c0659', '0522c0661',
    '0522c0667', '0522c0669', '0522c0708', '0522c0727', '0522c0746',]

        
    test_data_list = []
    test_label_list = []
    test_center_list = []

    for name in test_name_list:
        index_name = name.split('_')[0]

        CT = sitk.ReadImage(data_path + '/%s'%name + '_data.nii.gz')
        label = sitk.ReadImage(data_path + '/%s_label.nii.gz'%index_name)

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
    small_dice_list = np.zeros(3)
    small_index = [2,4,5]

    for i in range(len(test_data_list)):
        tmp_dice_list = np.zeros(9)
        tmp_small_dice_list = np.zeros(3)

        itkCT = test_data_list[i]
        gt_center = test_center_list[i]

        itkLabel = test_label_list[i]

        itkPred, itksmallPred, pred_center = evaluation(net, itkCT, SMALL=True)

        tmp_distance_list = cal_distance(pred_center, gt_center, itkCT)

        for idx in range(1, 10):
            dicecomputer.Execute(itkLabel==idx, itkPred==idx)
            dice = dicecomputer.GetDiceCoefficient()
            tmp_dice_list[idx-1] += dice
        for idx in range(3):
            dicecomputer.Execute(itkLabel==small_index[idx], itksmallPred==small_index[idx])
            dice = dicecomputer.GetDiceCoefficient()
            tmp_small_dice_list[idx] += dice


        print('dice', tmp_dice_list.mean(), 'distance', tmp_distance_list.mean())
        print('small dice', tmp_small_dice_list.mean(), tmp_small_dice_list)
        small_dice_list += tmp_small_dice_list

        dice_list += tmp_dice_list
        distance_list += tmp_distance_list

    dice_list /= len(test_data_list)
    small_dice_list /= len(test_data_list)
    distance_list /= len(test_data_list)

    return dice_list, small_dice_list, distance_list



if __name__ == '__main__':


    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=300, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=1,
            type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default='./checkpoint/heatmap_best.pth',
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path',
            default='./log/', help='log path')
    parser.add_option('--data_path', type='str', dest='data_path',
            default='/research/cbim/vast/yg397/OAR/dataset/MICCAI2015_dataset/', help='data_path')
    
    parser.add_option('-m', type='str', dest='model',
            default='focusnet', help='use which model')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='adv_test', help='use which model')
    parser.add_option('--norm', type='str', dest='norm', default='bn')
    parser.add_option('--gpu', type='str', dest='gpu',
            default='3')

    np.set_printoptions(precision=5)
    (options, args) = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    
    pdb.set_trace()
    print('use model:', options.model)
    if options.model == 'focusnet':
        net = focus_net(1, 10, se=True, norm=options.norm, SOSNet=True)


    AE_path = './checkpoint/MICCAI_AE_best.pth'
    AENet = ConvAE(1, 1, 512)
    AENet = nn.DataParallel(AENet)
    #AENet = load_my_state_dict(AENet, torch.load(AE_path))
    pth = torch.load(AE_path)
    AENet.load_state_dict(pth, strict=True)
    print('AE loaded from {}'.format(AE_path))
    
    if options.load:
        net = load_my_state_dict(net, torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    net.cuda()
    AENet.cuda()
    train_net(net, AENet, options)

    print('done')

