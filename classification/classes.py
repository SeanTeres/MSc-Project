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
import torch.nn.functional as F
import torch.nn as nn

from helpers import read_and_normalize_xray, split_with_indices

resize_transform = transforms.Compose([xrv.datasets.XRayResizer(224)])

class DICOMDataset1(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=(224, 224)):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.metadata_df)


    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['(0020,000d) UI Study Instance UID']
        profusion = self.metadata_df.iloc[idx]['Profusion']


        if profusion in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']:
            profusion_label = 1
        else:
            profusion_label = 0

        dicom_file = os.path.join(self.dicom_dir, dicom_filename + '.dcm')

        pixel_tensor, pixel_array = read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Return the processed image and label information
        return pixel_tensor, profusion_label
    
    
class DICOMDataset2(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=(224, 224)):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.metadata_df)


    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['Anonymized Filename']
        profusion = self.metadata_df.iloc[idx]['Radiologist: Silicosis (Profusion ≥ 1/1)']

        if profusion:
            profusion_label = 1
        else:
            profusion_label = 0

        dicom_file = os.path.join(self.dicom_dir, dicom_filename) # remove .dcm for D2

        pixel_tensor, pixel_array = read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Return the processed image and label information
        return pixel_tensor, profusion_label
    
class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmentations_list):
        """
        base_dataset: Original dataset.
        augmentations_list: List of transformations to apply individually to each sample.
        """
        self.base_dataset = base_dataset
        self.augmentations_list = augmentations_list

    def __len__(self):
        # Each original sample is repeated once for each augmentation plus the original
        return len(self.base_dataset) * (len(self.augmentations_list) + 1)

    def __getitem__(self, idx):
        # Identify the original sample and which augmentation (if any) to apply
        original_idx = idx // (len(self.augmentations_list) + 1)
        augment_idx = idx % (len(self.augmentations_list) + 1)

        pixel_tensor, label = self.base_dataset[original_idx]

        if augment_idx > 0:  # Apply the specific augmentation
            augmentation = self.augmentations_list[augment_idx - 1]
            pixel_tensor = augmentation(pixel_tensor)

        return pixel_tensor, label


class BaseClassifier1(nn.Module):
    def __init__(self, in_features):
        super(BaseClassifier1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 512)  # Input size is 1024
        self.fc2 = nn.Linear(512, 256)            # Additional hidden layer
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer
        return x