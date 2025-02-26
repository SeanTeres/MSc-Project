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

import helpers

class DICOMDataset1_old(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None
        
        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label",
        }
    
    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")
        
        self.target_label = target_label
        self.target_size = target_size
        
        # Pre-compute all labels
        self._assign_labels()
    
    def _assign_labels(self):
        for idx in range(len(self.metadata_df)):
            prof_label = self.metadata_df.iloc[idx]['Profusion']
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']
            
            tba_1_bool = helpers.contains_tba_or_tbu(tba_1)
            tba_2_bool = helpers.contains_tba_or_tbu(tba_2)
            
            # Profusion Label
            self.metadata_df.loc[idx, 'Profusion Label'] = 1 if prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3'] else 0
            
            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if (tba_1_bool or tba_2_bool) else 0
            
            # Profusion or TBA/TBU Label
            prof_positive = prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if (prof_positive or (tba_1_bool or tba_2_bool)) else 0
            
            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if (prof_positive and (tba_1_bool or tba_2_bool)) else 0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['(0020,000d) UI Study Instance UID']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename + '.dcm')

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])
        
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        
        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, target

class DICOMDataset1(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None
        
        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label",
        }
    
    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")
        
        self.target_label = target_label
        self.target_size = target_size
        
        # Pre-compute all labels
        self._assign_labels()
    
    def _assign_labels(self):
        for idx in range(len(self.metadata_df)):
            prof_label = self.metadata_df.iloc[idx]['Profusion']
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']
            
            tba_1_bool = helpers.contains_tba_or_tbu(tba_1)
            tba_2_bool = helpers.contains_tba_or_tbu(tba_2)
            
            # Profusion Label
            self.metadata_df.loc[idx, 'Profusion Label'] = 1 if prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3'] else 0
            
            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if (tba_1_bool and tba_2_bool) else 0
            
            # Profusion or TBA/TBU Label
            prof_positive = prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if (prof_positive or (tba_1_bool and tba_2_bool)) else 0
            
            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if (prof_positive and (tba_1_bool and tba_2_bool)) else 0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['(0020,000d) UI Study Instance UID']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename + '.dcm')

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])
        
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        
        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, target

class DICOMDataset2(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None
        
        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label"
        }
    
    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")
        
        self.target_label = target_label
        self.target_size = target_size
        
        # Pre-compute all labels
        self._assign_labels()
    
    def _assign_labels(self):
        for idx in range(len(self.metadata_df)):
            profusion = self.metadata_df.iloc[idx]['Radiologist: Silicosis (Profusion ≥ 1/1)']
            tba = self.metadata_df.iloc[idx]['Radiologist: TB (TBA or TBU)']
            
            # Profusion Label
            self.metadata_df.loc[idx, 'Profusion Label'] = 1 if profusion else 0
            
            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if tba else 0
            
            # Profusion or TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if (profusion or tba) else 0
            
            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if (profusion and tba) else 0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['Anonymized Filename']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename)

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])
        
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

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

        pixel_tensor = (pixel_tensor - pixel_tensor.min()) / (pixel_tensor.max() - pixel_tensor.min())
        # Rescale to [-1024, 1024] if needed for xrv models
        pixel_tensor = pixel_tensor * (1024 - (-1024)) + (-1024)

        return pixel_tensor, label


class BaseClassifier(nn.Module):
    def __init__(self, in_features):
        super(BaseClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 512)  # Input size is 1024
        self.fc2 = nn.Linear(512, 256)            # Additional hidden layer
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer
        return x
    
class BaseClassifierWithDropout(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):
        super(BaseClassifierWithDropout, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 512)  # Input size is 1024
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)  # Additional hidden layer
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.dropout2(x)
        x = self.fc3(x)  # Output layer
        return x

class BaseClassifier512(nn.Module):
    def __init__(self, in_features):
        super(BaseClassifier512, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 1024)  # Input size is 1024
        self.fc2 = nn.Linear(1024, 512)            # Additional hidden layer
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x
    
class PNGDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        
        
        label = image_path[len(image_path) - 5]
        image = (image - image.min()) / (image.max() - image.min())
        # Rescale to [-1024, 1024] if needed for xrv models
        image = image * (1024 - (-1024)) + (-1024)
        return image, label