import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
import pydicom
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
import wandb

from classification.helpers import read_and_normalize_xray, split_with_indices, create_dataloaders
from classification.classes import DICOMDataset1, DICOMDataset2, AugmentedDataset, BaseClassifier1
from classification.train_utils import train_model, test_model

dicom_dir_1 = 'MBOD_Datasets/Dataset 1'
metadata_1 = pd.read_excel('MBOD_Datasets/Dataset 1/FileDatabaseWithRadiology.xlsx')

dicom_dir_2 = 'MBOD_Datasets/Dataset 2'
metadata_2 = pd.read_excel('MBOD_Datasets/Dataset 2/Database_Training-2024.08.28.xlsx')


d1 = DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1)

d2 = DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2)

train_d1, val_d1, test_d1 = split_with_indices(d1, 0.7)
train_d2, val_d2, test_d2 = split_with_indices(d2, 0.7)

# Define augmentations as individual transforms
augmentations_list = [
    transforms.RandomHorizontalFlip(p=1.0),  # Always flip
    transforms.RandomRotation(15),# Rotate by 15 degrees
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) # Slight Blur

]

# Wrap the training set in the augmented dataset
augmented_train_d1 = AugmentedDataset(base_dataset=train_d1, augmentations_list=augmentations_list)
augmented_train_d2 = AugmentedDataset(base_dataset=train_d2, augmentations_list=augmentations_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = xrv.models.DenseNet(weights="densenet121-res224-all")
model_1 = model_1.to(device)

in_features = 1024  # Based on the output of model.features2()
model_1.classifier = BaseClassifier1(in_features).to(device)

train_loader_d1, train_aug_loader_d1, val_loader_d1, test_loader_d1 = create_dataloaders(train_d1, augmented_train_d1, val_d1, test_d1, 32)
train_loader_d2, train_aug_loader_d2, val_loader_d2, test_loader_d2 = create_dataloaders(train_d2, augmented_train_d2, val_d2, test_d2, 32)

def read_data(d1, d2):
    train_d1, val_d1, test_d1 = split_with_indices(d1, 0.7)
    train_d2, val_d2, test_d2 = split_with_indices(d2, 0.7)

    # Define augmentations as individual transforms
    augmentations_list = [
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.RandomRotation(15),# Rotate by 15 degrees
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) # Slight Blur

    ]

    # Wrap the training set in the augmented dataset
    augmented_train_d1 = AugmentedDataset(base_dataset=train_d1, augmentations_list=augmentations_list)
    augmented_train_d2 = AugmentedDataset(base_dataset=train_d2, augmentations_list=augmentations_list)

    return train_d1, augmented_train_d1, val_d1, test_d1, train_d2, augmented_train_d2, val_d2, test_d2

