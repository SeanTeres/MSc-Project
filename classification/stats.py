import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
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

def plot_all_label_distributions(df, dataset_name):
    label_columns = [
        'Profusion Label',
        'TBA/TBU Label', 
        'Profusion or TBA/TBU Label',
        'Profusion and TBA/TBU Label'
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

d1_prof = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1, target_size=224, target_label='Profusion')
d2_prof = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2, target_size=224, target_label='Profusion')

# Plot distributions for Dataset 1
# print(f"Dataset 1 Distributions:")
# plot_all_label_distributions(d1_prof.metadata_df, "MBOD 1")

# Plot distributions for Dataset 2
# print("\nDataset 2 Distributions:")
# plot_all_label_distributions(d2_prof.metadata_df, "MBOD 2")

train_d1, val_d1, test_d1 = helpers.split_with_indices(d2_prof, 0.7)

train_labels_d1 = d2_prof.metadata_df.loc[train_d1.indices, 'Profusion Label']

sample_weights_d1 = helpers.calculate_sample_weights(train_labels_d1)

print(f"Sample Weights for Dataset 2 (Profusion): {np.unique(sample_weights_d1)}")
print(d2_prof.metadata_df['Profusion Label'].value_counts())

