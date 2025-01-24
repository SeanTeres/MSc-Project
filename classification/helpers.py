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
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn

def read_and_normalize_xray(dicom_name, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True):
    """Reads a DICOM file, normalizes it, and returns the tensor and pixel array."""
    ds = pydicom.dcmread(dicom_name)

    if voi_lut:
        pixel_array = pydicom.apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
    else:
        pixel_array = ds.pixel_array.astype(np.float32)

    if ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        pixel_array = rgb2gray(pixel_array)

    if fix_monochrome and ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.amax(pixel_array) - pixel_array

    pixel_array = pixel_array.astype(np.float32)
    # Convert to tensor (1, H, W) and apply transforms (resize, crop)
    pixel_tensor = torch.from_numpy(pixel_array).unsqueeze(0)  # Add channel dimension
    if transforms:
        pixel_tensor = transforms(pixel_tensor)

    # Normalize if specified
    if normalize:
        pixel_tensor = (pixel_tensor - pixel_tensor.min()) / (pixel_tensor.max() - pixel_tensor.min())
        # Rescale to [-1024, 1024] if needed for xrv models
        pixel_tensor = pixel_tensor * (1024 - (-1024)) + (-1024)

    pixel_array = pixel_tensor.numpy()

    return pixel_tensor, pixel_array


def split_with_indices(dataset_class, train_size, to_augment=False):
    """Takes a dataset class and the training set size, validation and test splits are even for the remaining portion."""
    test_set_size = 1 - train_size
    indices_d =  list(range(len(dataset_class)))

    train_indices, test_indices = train_test_split(indices_d, test_size=test_set_size, random_state=42)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset_class, train_indices)
    val_dataset = Subset(dataset_class, val_indices)
    test_dataset = Subset(dataset_class, test_indices)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train, aug_train, val, test, batch_size):
    """Function to create dataloaders for different sets (incl. augmented) based on batch size.
    Only shuffles for train and aug_train sets."""
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    aug_train_loader = DataLoader(aug_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return train_loader, aug_train_loader, val_loader, test_loader

