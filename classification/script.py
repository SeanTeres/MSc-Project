import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
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

# Initialize datasets
d1 = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d2 = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2)

train_d1, augmented_train_d1, val_d1, test_d1, train_d2, augmented_train_d2, val_d2, test_d2 = helpers.read_data(d1, d2)

for experiment_name, experiment in config['experiments'].items():
    print(f"Running experiment: {experiment_name}")
    lr = experiment['lr']
    n_epochs = experiment['n_epochs']
    batch_size = experiment['batch_size']
    train_dataset = experiment['train_dataset']
    model = experiment['model']
    target = experiment['target_label']
    resolution = experiment['resolution']
    oversampling = experiment['oversampling']

    print(f"MODEL :{model}")
    # Set target and resolution for datasets
    d1.set_target(target, resolution)
    d2.set_target(target, resolution)

    # Define augmentations as individual transforms
    augmentations_list = [
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.RandomRotation(15),  # Rotate by 15 degrees
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # Slight Blur
    ]
    
    augmented_train_d1 = classes.AugmentedDataset(base_dataset=train_d1, augmentations_list=augmentations_list)
    augmented_train_d2 = classes.AugmentedDataset(base_dataset=train_d2, augmentations_list=augmentations_list)

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)

    in_features = 1024  # Based on the output of model.features2()
    model.classifier = classes.BaseClassifier(in_features).to(device)

    # Create dataloaders
    train_loader_d1, train_aug_loader_d1, val_loader_d1, test_loader_d1 = helpers.create_dataloaders(
        train_d1, augmented_train_d1, val_d1, test_d1, experiment['batch_size'], oversam=experiment['oversampling']
    )

    train_loader_d2, train_aug_loader_d2, val_loader_d2, test_loader_d2 = helpers.create_dataloaders(
        train_d2, augmented_train_d2, val_d2, test_d2, experiment['batch_size'], oversam=experiment['oversampling']
    )
    # Log oversampling information
    if oversampling:
        print("We are oversampling!!")

    # Print label distribution
    print("Label distribution for training set (D1):", helpers.calc_label_dist(d1, train_loader_d1.dataset, target + ' Label'))
    print("Label distribution for validation set (D1):", helpers.calc_label_dist(d1, val_loader_d1.dataset, target + ' Label'))
    print("Label distribution for test set (D1):", helpers.calc_label_dist(d1, test_loader_d1.dataset, target + ' Label'))
    
    print("Label distribution for training set (D2):", helpers.calc_label_dist(d2, train_loader_d2.dataset, target + ' Label'))
    print("Label distribution for validation set (D2):", helpers.calc_label_dist(d2, val_loader_d2.dataset, target + ' Label'))
    print("Label distribution for test set (D2):", helpers.calc_label_dist(d2, test_loader_d2.dataset, target + ' Label'))

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
        model = train_utils.train_model(train_loader_d1, val_loader_d1, model, n_epochs, lr, device)
    else:
        print("Training on Dataset 2\n")
        model = train_utils.train_model(train_loader_d2, val_loader_d2, model, n_epochs, lr, device)

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

    print(f"Confusion Matrix ({experiment_name} - MBOD 2):")
    print(cm_d2)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm_d1, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({experiment_name})- MBOD 1)")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_d2, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({experiment_name}) - MBOD 2)")

    plt.savefig(f"new_plots/conf_mat_{experiment_name}.png")

    wandb.finish()