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


class DICOMDataset1(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224, target_label=None):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = target_label

    def __len__(self):
        return len(self.metadata_df)


    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['(0020,000d) UI Study Instance UID']

        if (self.target_label == "Profusion"):
            profusion = self.metadata_df.iloc[idx]['Profusion']

            if profusion in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']:
                target = 1
            else:
                target = 0

        elif(self.target_label == "TBA/TBU"):
            
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']

            # Check if 'tba' or 'tbu' is in tba_1 or tba_2
            contains_tba_or_tbu_1 = 'tba' in tba_1 or 'tbu' in tba_1
            contains_tba_or_tbu_2 = 'tba' in tba_2 or 'tbu' in tba_2

            # Example usage
            if contains_tba_or_tbu_1 or contains_tba_or_tbu_2:
                target = 1
            else:
                target = 0

        elif(self.target_label == "Profusion Or TBA/TBU"):
            profusion = self.metadata_df.iloc[idx]['Profusion']
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']

            # Check if 'tba' or 'tbu' is in tba_1 or tba_2
            contains_tba_or_tbu_1 = 'tba' in tba_1 or 'tbu' in tba_1
            contains_tba_or_tbu_2 = 'tba' in tba_2 or 'tbu' in tba_2

            # Example usage
            if (profusion in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']) or contains_tba_or_tbu_1 or contains_tba_or_tbu_2:
                target = 1
            else:
                target = 0
        
        elif (self.target_label == "Profusion and TBA/TBU"):
            profusion = self.metadata_df.iloc[idx]['Profusion']
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']

            contains_tba_or_tbu_1 = 'tba' in tba_1 or 'tbu' in tba_1
            contains_tba_or_tbu_2 = 'tba' in tba_2 or 'tbu' in tba_2

            if (profusion in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']) and (contains_tba_or_tbu_1 or contains_tba_or_tbu_2):
                target = 1
            else:
                target = 0
        else:
            print("Target not recognized")

        dicom_file = os.path.join(self.dicom_dir, dicom_filename + '.dcm')

        pixel_tensor, pixel_array = read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])

        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Return the processed image and label information
        return pixel_tensor, target
    
    
class DICOMDataset2(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224, target_label=None):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = target_label

    def __len__(self):
        return len(self.metadata_df)


    def __getitem__(self, idx):

        dicom_filename = self.metadata_df.iloc[idx]['Anonymized Filename']

        if (self.target_label == "Profusion"):
            profusion = self.metadata_df.iloc[idx]['Radiologist: Silicosis (Profusion ≥ 1/1)']

            target = 1 if profusion else 0

        elif(self.target_label == "TBA/TBU"):
            tba = self.metadata_df.iloc[idx]['Radiologist: TB (TBA or TBU)']

            target = 1 if tba else 0

        elif(self.target_label == "Profusion Or TBA/TBU"):

            profusion = self.metadata_df.iloc[idx]['Radiologist: Silicosis (Profusion ≥ 1/1)']
            tba = self.metadata_df.iloc[idx]['Radiologist: TB (TBA or TBU)']

            target = 1 if (profusion or tba) else 0
        
        elif(self.target_label == "Profusion and TBA/TBU"):

            profusion = self.metadata_df.iloc[idx]['Radiologist: Silicosis (Profusion ≥ 1/1)']
            tba = self.metadata_df.iloc[idx]['Radiologist: TB (TBA or TBU)']

            target = 1 if (profusion and tba) else 0

        dicom_file = os.path.join(self.dicom_dir, dicom_filename) # remove .dcm for D2

        pixel_tensor, pixel_array = read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)

        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])

        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Return the processed image and label information
        return pixel_tensor, target
    
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


class BaseClassifier(nn.Module):
    def __init__(self, in_features):
        super(BaseClassifier, self).__init__()
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