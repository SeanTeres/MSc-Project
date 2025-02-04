import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, Subset, DataLoader
import os
import torchxrayvision as xrv
import torchvision.transforms as transforms
from skimage.color import rgb2gray
from skimage.transform import resize
import pydicom
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
import wandb
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import helpers, train_utils, classes
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim


def plot_all_label_distributions(df, dataset_name):
    label_columns = [
        'Profusion Label',
        'TBA-TBU Label', 
        'Profusion or TBA-TBU Label',
        'Profusion and TBA-TBU Label'
    ]
    colors = ['#ff9999', '#66b3ff']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, column in enumerate(label_columns):
        value_counts = df[column].value_counts().sort_index()
        percentages = df[column].value_counts(normalize=True).sort_index() * 100
        
        bars = axes[idx].bar(range(len(value_counts)), value_counts.values, color=colors[:len(value_counts)])
        axes[idx].set_title(f'{column} Distribution - {dataset_name}')
        axes[idx].set_xlabel('Label (0: Negative, 1: Positive)')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(['0', '1'])
        
        # Add legend to each subplot
        axes[idx].legend(bars, ['Negative', 'Positive'], 
                        loc='upper left')
        
        # Updated label positioning
        for i, (count, percentage) in enumerate(zip(value_counts, percentages)):
            height = value_counts.values[i]
            axes[idx].text(i, height/2, f'{count}\n({percentage:.1f}%)', 
                         ha='center', va='center')
        
        print(f"\nDistribution Summary for {column} - {dataset_name}:")
        print(f"Counts and Percentages:")
        for label, count, pct in zip(['Negative (0)', 'Positive (1)'], 
                                   value_counts, percentages):
            print(f"{label}: {count} ({pct:.1f}%)")
    
    plt.tight_layout()
    plt.savefig(f"binary_dist_{dataset_name}.png", bbox_inches='tight')
    plt.show()


dicom_dir_1 = 'MBOD_Datasets/Dataset 1'
metadata_1 = pd.read_excel('MBOD_Datasets/Dataset 1/FileDatabaseWithRadiology.xlsx')

dicom_dir_2 = 'MBOD_Datasets/Dataset 2'
metadata_2 = pd.read_excel('MBOD_Datasets/Dataset 2/Database_Training-2024.08.28.xlsx')

d1 = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1, target_size=224) 
d2 = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2, target_size=224)

# Split datasets and store indices
train_indices_d1, val_indices_d1, test_indices_d1 = helpers.split_dataset(d1)
train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2)

# Save indices for later use
split_indices = {
    'd1': {'train': train_indices_d1, 'val': val_indices_d1, 'test': test_indices_d1},
    'd2': {'train': train_indices_d2, 'val': val_indices_d2, 'test': test_indices_d2}
}

label = 'Profusion and TBA-TBU'
d1.set_target(target_label=label, target_size=224)
d2.set_target(target_label=label, target_size=224)

train_d1 = Subset(d1, train_indices_d1)
val_d1 = Subset(d1, val_indices_d1)
test_d1 = Subset(d1, test_indices_d1)

train_d2 = Subset(d2, train_indices_d2)
val_d2 = Subset(d2, val_indices_d2)
test_d2 = Subset(d2, test_indices_d2)

def create_weighted_sampler(dataset, target_label):
    # Calculate class weights
    class_counts = np.bincount([label for _, label in dataset])
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for _, label in dataset]

    # Create a weighted sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

# Create the base datasets
train_d1 = Subset(d1, train_indices_d1)
train_d2 = Subset(d2, train_indices_d2)

# Define augmentations
augmentations_list = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
]

# Create augmented datasets
augmented_train_d1 = classes.AugmentedDataset(base_dataset=train_d1, augmentations_list=augmentations_list)
augmented_train_d2 = classes.AugmentedDataset(base_dataset=train_d2, augmentations_list=augmentations_list)

    # Create dataloaders
train_loader_d1, train_aug_loader_d1, val_loader_d1, test_loader_d1 = helpers.create_dataloaders(
    train_d1, augmented_train_d1, val_d1, test_d1, batch_size=32, oversam=True, target=label
)

train_loader_d2, train_aug_loader_d2, val_loader_d2, test_loader_d2 = helpers.create_dataloaders(
    train_d2, augmented_train_d2, val_d2, test_d2, batch_size=32, oversam=True, target=label
)



# Plot distributions for Dataset 1
# print(f"Dataset 1 Distributions:")
# plot_all_label_distributions(d1.metadata_df, "MBOD 1")

# Plot distributions for Dataset 2
# print("\nDataset 2 Distributions:")
# plot_all_label_distributions(d2.metadata_df, "MBOD 2")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ae = xrv.autoencoders.ResNetAE(weights="101-elastic").to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

mc_folder = 'C:/Users/user-pc/Masters/MSc - Project/Intl_Datasets/MC-png'
sz_folder = 'C:/Users/user-pc/Masters/MSc - Project/Intl_Datasets/SZ-png'

mc_dataset = classes.PNGDataset(mc_folder, transform=transform)
sz_dataset = classes.PNGDataset(sz_folder, transform=transform)


mc_loader = DataLoader(mc_dataset, batch_size=32, shuffle=False)
sz_loader = DataLoader(sz_dataset, batch_size=32, shuffle=False)


x = DataLoader(train_d1, batch_size=32, shuffle=False)
y = DataLoader(augmented_train_d1, batch_size=32, shuffle=False)

print(len(x), len(y))
