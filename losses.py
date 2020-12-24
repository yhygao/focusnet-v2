import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb 

class DiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)

        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.) 

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P 
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.alpha * torch.sum(FP, dim=(0)).float() + self.beta * torch.sum(FN, dim=(0)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss
        loss = 1 - dice
        if weight is not False:
            loss *= weight.squeeze(0)
        loss = loss.sum()
        if self.size_average:
            if weight is not False:
                loss /= weight.squeeze(0).sum()
            else:
                loss /= C

        return loss
class BinaryDiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(BinaryDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets):
        N = preds.size(0)
        
        preds = F.sigmoid(preds)
        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)
        P = preds
        targets = targets.view(-1, 1)

        smooth = torch.zeros(1, dtype=torch.float32).fill_(0.00001)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - targets
        class_mask = targets

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.alpha * torch.sum(FP, dim=(0)).float() + self.beta * torch.sum(FN, dim=(0)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss


        dice = dice.sum()
        loss = 1. - dice

        return loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.Tensor([0.5]).cuda()
        else:
            self.alpha = torch.Tensor([alpha]).cuda()

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets):
        N = preds.size(0)
        targets = targets.long()

        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1)
        targets = targets.view(-1)

        P = F.sigmoid(preds)
        log_P = F.logsigmoid(preds)
        log_P_ = F.logsigmoid(1 - preds)

        targets = targets.float()

        batch_loss = -self.alpha * (1-P).pow(self.gamma)*log_P * targets \
                    -(1-self.alpha) * P.pow(self.gamma)*log_P_ * (1-targets)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)

        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        P = F.softmax(preds, dim=1)
        log_P = F.log_softmax(preds, dim=1)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.)

        alpha = self.alpha[targets.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_probs = (log_P * class_mask).sum(1).view(-1, 1)

        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs
        if weight is not False:
            element_weight = weight.squeeze(0)[targets.squeeze(0)]
            batch_loss = batch_loss * element_weight

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

