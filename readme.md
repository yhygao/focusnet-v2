# Focusnet
A pytorch implementation of the paper "FocusNetv2: Imbalanced large and small organ segmentation with adversarial shape constraint for head and neck CT images" and "Focusnet: Imbalanced large and small organ segmentation with an end-to-end deep neural network for head and neck ct images"

##Introduction


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

