import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, ConcatDataset
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
from collections import Counter
import classes
from tqdm import tqdm

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

def split_dataset(dataset, train_size=0.7, random_state=42):
    """Split dataset into train, validation, and test sets with fixed indices."""
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=1-train_size, random_state=random_state)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=random_state)
    
    return train_indices, val_indices, test_indices

def create_dataloaders(train, aug_train, val, test, batch_size, target):
    """Function to create dataloaders with optional oversampling.
    target: full string of label."""
    # print("Creating dataloaders with optional oversampling...")
    print(f"LENGTHS: {len(train), len(aug_train)}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    aug_train_loader = DataLoader(aug_train, batch_size=batch_size, shuffle=True)
    
    # Other loaders remain the same
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(f"LENGTHS: {len(train_loader), len(aug_train_loader)}")
    
    return train_loader, aug_train_loader, val_loader, test_loader

def contains_tba_or_tbu(string):
    if isinstance(string, float):
        return False
    return 'tba' in string.lower() or 'tbu' in string.lower()

def assign_labels_to_d1(d1):

    for i in range(len(d1)):
        prof_label = d1.metadata_df.iloc[i]['Profusion']

        tba_1 = d1.metadata_df.iloc[i]['strFindingsSimplified1']
        tba_2 = d1.metadata_df.iloc[i]['strFindingsSimplified2']

        if prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']:
            d1.metadata_df.at[i, 'Prof Label'] = True
        else:
            d1.metadata_df.at[i, 'Prof Label'] = False
        
        if contains_tba_or_tbu(tba_1) or contains_tba_or_tbu(tba_2):
            d1.metadata_df.at[i, 'TBA/TBU Label'] = True
        else:
            d1.metadata_df.at[i, 'TBA/TBU Label'] = False
        
        if (prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']) or (contains_tba_or_tbu(tba_1) or contains_tba_or_tbu(tba_2)):
            d1.metadata_df.at[i, 'Profusion Or TBA/TBU Label'] = True
        else:
            d1.metadata_df.at[i, 'Profusion Or TBA/TBU Label'] = False
        
        if (prof_label in ['1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']) and (contains_tba_or_tbu(tba_1) or contains_tba_or_tbu(tba_2)):
            d1.metadata_df.at[i, 'Profusion Or TBA/TBU Label'] = True
        else:
            d1.metadata_df.at[i, 'Profusion Or TBA/TBU Label'] = False
    
    return d1

def assign_labels_to_d2(d2):
    for i in range(len(d2)):

        prof = d2.metadata_df.ilov[i]['Radiologist: Silicosis (Profusion ≥ 1/1)']

        tba = d2.metadata_df.iloc[i]['Radiologist: TB (TBA or TBU)']

        if (prof):
            d2.metadata_df.at[i, 'Prof Label'] = True
        else:
            d2.metadata_df.at[i, 'Prof Label'] = False

        if (tba):
            d2.metadata_df.at[i, 'TBA/TBU Label'] = True
        else:
            d2.metadata_df.at[i, 'TBA/TBU Label'] = False
        
        if (prof or tba):
            d2.metadata_df.at[i, 'Profusion Or TBA/TBU Label'] = True
        else:
            d2.metadata_df.at[i, 'Profusion Or TBA/TBU Label'] = False
        
        if (prof and tba):
            d2.metadata_df.at[i, 'Profusion and TBA/TBU Label'] = True
        else:
            d2.metadata_df.at[i, 'Profusion and TBA/TBU Label'] = False
    
    return d2

# Function to calculate sample weights for WeightedRandomSampler
def calculate_sample_weights(labels):
    """
    Calculate sample weights for a dataset to enable oversampling using WeightedRandomSampler.
    Args:
        labels (pd.Series): Class labels for the dataset.
    Returns:
        np.ndarray: Sample weights for each data point.
    """
    class_counts = labels.value_counts().to_dict()  # Count occurrences of each class
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = labels.map(class_weights).to_numpy()  # Map weights to each sample
    return sample_weights

def calc_label_dist(dataset, subset, disease_label):
    """Calculates label distribution for a dataset or subset at specified disease label."""
    if len(dataset) == len(subset):
        # Full dataset case
        labels = dataset.metadata_df[disease_label]
    else:
        # Subset case
        labels = dataset.metadata_df.loc[subset.indices, disease_label]
    
    # Convert labels to integers
    labels = labels.astype(int)
    
    return Counter(labels)

from tqdm import tqdm

def extract_pixel_intensities(dataloader):
    pixel_intensities = []
    for images, _ in tqdm(dataloader, desc="Extracting pixel intensities"):
        for image in images:
            pixel_intensities.extend(image.numpy().flatten())
    return pixel_intensities

import numpy as np

def get_alpha_FLoss(train, target_label):
    """
    Compute a single alpha value for binary focal loss.
    
    """
    class_counts = train.dataset.metadata_df[target_label].value_counts()
    class_counts = class_counts.sort_index()

    # Ensure class counts exist
    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both positive (1) and negative (0) samples.")

    minority = class_counts[0]  # Assuming class 0 is the minority
    majority = class_counts[1]  # Assuming class 1 is the majority

    # Compute alpha as the proportion of the negative class
    alpha = minority / (majority + minority)

    return alpha


def compute_pos_weight(train, target_label):
    """Compute pos_weight for BCEWithLogitsLoss."""
    class_counts = train.dataset.metadata_df[target_label].value_counts()
    class_counts = class_counts.sort_index()

    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both positive (1) and negative (0) samples.")

    N_pos = class_counts[1]  # Positive class count
    N_neg = class_counts[0]  # Negative class count

    pos_weight = torch.tensor([ N_neg / N_pos ], dtype=torch.float32)

    return pos_weight

def salt_and_pepper_noise_tensor(image, prob=0.02):
    """
    Apply salt-and-pepper noise to a PyTorch tensor image.
    
    :param image: PyTorch tensor of shape (C, H, W), values in [0,1].
    :param prob: Probability of a pixel being affected.
    :return: Noisy image tensor.
    """
    assert image.dim() == 3, "Input must be a 3D tensor (C, H, W)"
    
    noisy_image = image.clone()  # Clone to avoid modifying original image
    
    # Generate random noise mask
    rand_tensor = torch.rand_like(image)  # Random values between [0,1]

    # Apply Salt (white pixels)
    noisy_image[rand_tensor < prob / 2] = 1.0  # If image is in [0,1], use 255.0 for [0,255]

    # Apply Pepper (black pixels)
    noisy_image[rand_tensor > 1 - prob / 2] = 0.0

    return noisy_image
