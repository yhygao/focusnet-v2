# Focusnet
A pytorch implementation of the paper "FocusNetv2: Imbalanced large and small organ segmentation with adversarial shape constraint for head and neck CT images" and "Focusnet: Imbalanced large and small organ segmentation with an end-to-end deep neural network for head and neck ct images"

## Introduction

Radiotherapy is a treatment where radiation is used to eliminate cancer cells. The delineation of organs-at-risk (OARs) is a vital step in radiotherapy treatment planning to avoid damage to healthy organs. For nasopharyngeal cancer, more than 20 OARs are needed to be precisely segmented in advance. The challenge of this task lies in complex anatomical structure, low-contrast organ contours, and the extremely imbalanced size between large and small organs. Common segmentation methods that treat them equally would generally lead to inaccurate small-organ labeling. We propose a novel two-stage deep neural network, FocusNetv2, to solve this challenging problem by automatically locating, ROI-pooling, and segmenting small organs with specifically designed small-organ localization and segmentation sub-networks while maintaining the accuracy of large organ segmentation. In addition to our original FocusNet, we employ a novel adversarial shape constraint on small organs to ensure the consistency between estimated small-organ shapes and organ shape prior knowledge. Our proposed framework is extensively tested on both self-collected dataset of 1,164 CT scans and the *MICCAI Head and Neck Auto Segmentation Challenge 2015* dataset, which shows superior performance compared with state-of-the-art head and neck OAR segmentation methods.

<div align=center>

<img src="https://raw.githubusercontent.com/yhygao/focusnet-v2/main/fig/framework.PNG" />

</div>

## Getting Started
#### Prerequisites
```
Python >= 3.6
pytorch = 1.0.1.post2
SimpleITK = 1.2.0
scikit-image = 0.16.2
```


#### Training
Prepare training dataset using code in data_preprocess.


Training the S-Net using train_backbone.py

For trianing SOL-Net, if you already have a trained S-Net, you can use train_SOL.py to train the localization network. If you don't have a trained S-Net, you can use train_backbone_heatmap.py to train S-Net and SOL-Net jointly.

To train the SOS-Net with shape constraint, pretrain the AutoEncoder first (training code not included yet). Then use adv_train.py to adversarially update the SOS-Net and the shape AutoEncoder.




## Citation

@article{gao2021focusnetv2,

title={FocusNetv2: Imbalanced large and small organ segmentation with adversarial shape constraint for head and neck CT images},

author={Gao, Yunhe and Huang, Rui and Yang, Yiwei and Zhang, Jie and Shao, Kainan and Tao, Changjuan and Chen, Yuanyuan and Metaxas, Dimitris N and Li, Hongsheng and Chen, Ming},

journal={Medical Image Analysis},

volume={67},

pages={101831},

year={2021},

publisher={Elsevier}

}

