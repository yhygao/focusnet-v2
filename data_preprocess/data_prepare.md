###Data Preparation

Download data from https://www.imagenglab.com/newsite/pddca/


Use Data_preprocess.py and crop_data.py to prepare training images. Data_preprocess.py is used to resample images and labels to the same spacing, which is (1, 1, 2.5) mm in our experiment. The crop_data.py is used for cropping the images to remove the black background.

The new_training.csv and new_test.csv contain the name of images for training and testing. As there are several images in the training set miss some structures, we indicate the presence of the structure using 1 and 0, and ignore the loss from the missing structure during training.