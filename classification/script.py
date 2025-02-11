import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, Subset
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

with open('classification/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dicom_dir_1 = 'MBOD_Datasets/Dataset 1'
metadata_1 = pd.read_excel('MBOD_Datasets/Dataset 1/FileDatabaseWithRadiology.xlsx')

dicom_dir_2 = 'MBOD_Datasets/Dataset 2'
metadata_2 = pd.read_excel('MBOD_Datasets/Dataset 2/Database_Training-2024.08.28.xlsx')

target_label = 'Profusion'
model_resolution = 224

# Initialize datasets
d1 = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d2 = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2)

d1.set_target(target_label, model_resolution)
d2.set_target(target_label, model_resolution)

# Split datasets and store indices
train_indices_d1, val_indices_d1, test_indices_d1 = helpers.split_dataset(d1)
train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2)

# Save indices for later use
split_indices = {
    'd1': {'train': train_indices_d1, 'val': val_indices_d1, 'test': test_indices_d1},
    'd2': {'train': train_indices_d2, 'val': val_indices_d2, 'test': test_indices_d2}
}


train_d1 = Subset(d1, train_indices_d1)
val_d1 = Subset(d1, val_indices_d1)
test_d1 = Subset(d1, test_indices_d1)

train_d2 = Subset(d2, train_indices_d2)
val_d2 = Subset(d2, val_indices_d2)
test_d2 = Subset(d2, test_indices_d2)

# Define augmentations
augmentations_list = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
]

# Create augmented datasets
augmented_train_d1 = classes.AugmentedDataset(base_dataset=train_d1, augmentations_list=augmentations_list)
augmented_train_d2 = classes.AugmentedDataset(base_dataset=train_d2, augmentations_list=augmentations_list)

for experiment_name, experiment in config['experiments'].items():
    print(f"Running experiment: {experiment_name}")
    lr = experiment['lr']
    n_epochs = experiment['n_epochs']
    batch_size = experiment['batch_size']
    train_dataset = experiment['train_dataset']
    model = experiment['model']
    oversampling = experiment['oversampling']
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    loss_function = experiment['loss_function']
    model = model.to(device)

    in_features = 1024  # Based on the output of model.features2()
    model.classifier = classes.BaseClassifier(in_features).to(device)


    # Create dataloaders
    train_loader_d1, train_aug_loader_d1, val_loader_d1, test_loader_d1 = helpers.create_dataloaders(
        train_d1, augmented_train_d1, val_d1, test_d1, experiment['batch_size'], oversam=experiment['oversampling'], target=target_label
    )

    train_loader_d2, train_aug_loader_d2, val_loader_d2, test_loader_d2 = helpers.create_dataloaders(
        train_d2, augmented_train_d2, val_d2, test_d2, experiment['batch_size'], oversam=experiment['oversampling'], target=target_label
    )

    
    # Calculate sample weights for weighted random sampling
    if oversampling:
        train_labels_d1 = [d1.metadata_df.iloc[idx][target_label + ' Label'] for idx in train_indices_d1]
        train_labels_d2 = [d2.metadata_df.iloc[idx][target_label + ' Label'] for idx in train_indices_d2]
        
        sample_weights_d1 = helpers.calculate_sample_weights(pd.Series(train_labels_d1))
        sample_weights_d2 = helpers.calculate_sample_weights(pd.Series(train_labels_d2))
        
        sampler_d1 = WeightedRandomSampler(sample_weights_d1, len(train_d1))
        sampler_d2 = WeightedRandomSampler(sample_weights_d2, len(train_d2))
    else:
        sampler_d1 = None
        sampler_d2 = None


    # Select the appropriate dataloaders based on the augmentation flag
    if experiment['augmentation']:
        print("We are using augmentations for training. \n")
        print(f"Batches before: {len(train_loader_d1) % 32}")
        print(f"Batches after: {len(train_aug_loader_d1) % 32}")
        selected_train_loader_d1 = train_aug_loader_d1
        selected_train_loader_d2 = train_aug_loader_d2
    else:
        selected_train_loader_d1 = train_loader_d1
        selected_train_loader_d2 = train_loader_d2

    # Print label distribution
    print("Label distribution for training set (D1):", helpers.calc_label_dist(d1, train_loader_d1.dataset, target_label + ' Label'))
    print("Label distribution for validation set (D1):", helpers.calc_label_dist(d1, val_loader_d1.dataset, target_label + ' Label'))
    print("Label distribution for test set (D1):", helpers.calc_label_dist(d1, test_loader_d1.dataset, target_label + ' Label'))
    
    print("Label distribution for training set (D2):", helpers.calc_label_dist(d2, train_loader_d2.dataset, target_label + ' Label'))
    print("Label distribution for validation set (D2):", helpers.calc_label_dist(d2, val_loader_d2.dataset, target_label + ' Label'))
    print("Label distribution for test set (D2):", helpers.calc_label_dist(d2, test_loader_d2.dataset, target_label + ' Label'))

    # Initialize wandb
    wandb.login()
    wandb.init(project='MBOD-2', name=experiment_name)
    wandb.config.update(experiment)

    in_features = 1024  # Based on the output of model.features2()
    model.classifier = classes.BaseClassifier(in_features).to(device)

    # Train and evaluate model
    test_labels_d1, test_labels_d2, test_preds_d1, test_preds_d2 = [], [], [], []

    if train_dataset == "MBOD 1":
        print("Training on Dataset 1\n")

        if(loss_function == "CrossEntropyLoss"):
            model = train_utils.train_model(selected_train_loader_d1, val_loader_d1, model, n_epochs, lr, device)

        elif (loss_function == "FocalLoss"):
            model = train_utils.train_model_with_focal_loss(selected_train_loader_d1, val_loader_d1, model, n_epochs, lr, device, alpha=0.25, gamma=1)
            
        else:
            print("ERR: Loss function must be CrossEntropyLoss or FocalLoss.")

    elif(train_dataset == "MBOD 2"):
        print("Training on Dataset 2\n")

        if(loss_function == "CrossEntropyLoss"):
            model = train_utils.train_model(selected_train_loader_d2, val_loader_d2, model, n_epochs, lr, device)

        elif(loss_function == "FocalLoss"):
            model = train_utils.train_model_with_focal_loss(selected_train_loader_d2, val_loader_d2, model, n_epochs, lr, device, alpha=0.25, gamma=1)
    else:
        print("ERR: Unrecognized dataset name.")

    test_labels_d1, test_preds_d1 = train_utils.test_model(test_loader_d1, model, device, "MBOD 1")
    test_labels_d2, test_preds_d2 = train_utils.test_model(test_loader_d2, model, device, "MBOD 2")

    # Generate and save confusion matrices
    report_d1 = classification_report(test_labels_d1, test_preds_d1)
    cm_d1 = confusion_matrix(test_labels_d1, test_preds_d1)

    report_d2 = classification_report(test_labels_d2, test_preds_d2)
    cm_d2 = confusion_matrix(test_labels_d2, test_preds_d2)

    print(f"Classification Report ({experiment_name}- MBOD 1)):")
    print(report_d1)

    print(f"Confusion Matrix ({experiment_name}- MBOD 1):")
    print(cm_d1)

    print("*" * 50)

    print(f"Classification Report ({experiment_name} - MBOD 2)")
    print(report_d2)

    print(f"{experiment_name} MBOD 1")
    print(cm_d2)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm_d1, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({experiment_name})- MBOD 1)")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_d2, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{experiment_name}) - MBOD 2")

    plt.savefig(f"new_plots/conf_mat_{experiment_name}.png")
    wandb.log({f"cm-{experiment_name}": wandb.Image(plt)})

    wandb.finish()