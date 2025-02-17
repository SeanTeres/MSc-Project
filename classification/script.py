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

# target_label = 'Profusion and TBA-TBU'
# model_resolution = 224

# Initialize datasets
d1 = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d2 = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2)

# Split datasets and store indices
train_indices_d1, val_indices_d1, test_indices_d1 = helpers.split_dataset(d1)
train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2)

# Save indices for later use
split_indices = {
    'd1': {'train': train_indices_d1, 'val': val_indices_d1, 'test': test_indices_d1},
    'd2': {'train': train_indices_d2, 'val': val_indices_d2, 'test': test_indices_d2}
}

# Define augmentations
augmentations_list = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10)
])


for experiment_name, experiment in config['experiments'].items():
    print(f"Running experiment: {experiment_name}")
    lr = experiment['lr']
    n_epochs = experiment['n_epochs']
    batch_size = experiment['batch_size']
    train_dataset = experiment['train_dataset']
    model = experiment['model']
    oversampling = experiment['oversampling']
    loss_function = experiment['loss_function']
    augmentations = experiment['augmentation']
    model_resolution = experiment['model_resolution']
    target_label = experiment['target']

    d1.set_target(target_label, model_resolution)
    d2.set_target(target_label, model_resolution)
    # Create augmented datasets
    d1_aug = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1, transform=augmentations_list)
    d2_aug = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2, transform=augmentations_list)

    d1_aug.set_target(target_label, model_resolution)
    d2_aug.set_target(target_label, model_resolution)

    train_d1 = Subset(d1, train_indices_d1)
    train_aug_d1 = Subset(d1_aug, train_indices_d1)

    val_d1 = Subset(d1, val_indices_d1)
    test_d1 = Subset(d1, test_indices_d1)

    train_d2 = Subset(d2, train_indices_d2)
    train_aug_d2 = Subset(d2_aug, train_indices_d2)

    val_d2 = Subset(d2, val_indices_d2)
    test_d2 = Subset(d2, test_indices_d2)

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)

    in_features = 1024  # Based on the output of model.features2()
    model.classifier = classes.BaseClassifier(in_features).to(device)


    # Create dataloaders
    train_loader_d1, train_aug_loader_d1, val_loader_d1, test_loader_d1 = helpers.create_dataloaders(
        train_d1, train_aug_d1, val_d1, test_d1, experiment['batch_size'], target=target_label
    )

    train_loader_d2, train_aug_loader_d2, val_loader_d2, test_loader_d2 = helpers.create_dataloaders(
        train_d2, train_aug_d2, val_d2, test_d2, experiment['batch_size'], target=target_label
    )        

    # Print label distribution
    print("Label distribution for training set (D1):", helpers.calc_label_dist(d1, train_loader_d1.dataset, target_label + ' Label'))
    print("Label distribution for validation set (D1):", helpers.calc_label_dist(d1, val_loader_d1.dataset, target_label + ' Label'))
    print("Label distribution for test set (D1):", helpers.calc_label_dist(d1, test_loader_d1.dataset, target_label + ' Label'))
    
    print("Label distribution for training set (D2):", helpers.calc_label_dist(d2, train_loader_d2.dataset, target_label + ' Label'))
    print("Label distribution for validation set (D2):", helpers.calc_label_dist(d2, val_loader_d2.dataset, target_label + ' Label'))
    print("Label distribution for test set (D2):", helpers.calc_label_dist(d2, test_loader_d2.dataset, target_label + ' Label'))

    # Initialize wandb
    wandb.login()
    wandb.init(project='MBOD-4', name=experiment_name)
    wandb.config.update(experiment)

    in_features = 1024  # Based on the output of model.features2()
    model.classifier = classes.BaseClassifier(in_features).to(device)

    # Train and evaluate model
    test_labels_d1, test_labels_d2, test_preds_d1, test_preds_d2 = [], [], [], []

    if augmentations:
        print(f"ON THE FLY AUGMENTATION!")
        train_loader_d1 = train_aug_loader_d1
        train_loader_d2 = train_aug_loader_d2
    else:
        print(f"NO AUGMENTATION!")
        train_loader_d1 = train_loader_d1
        train_loader_d2 = train_loader_d2

    if train_dataset == "MBOD 1":
        print("Training on Dataset 1\n")

        if(loss_function == "CrossEntropyLoss"):
            if(oversampling):
                pos_weight = helpers.compute_pos_weight(train_d1, target_label + ' Label')

                print(f"Oversampling with pos_weight = {pos_weight} ---- dataset {train_dataset}")
            else:
                pos_weight = torch.tensor([1.0])
            
            model = train_utils.train_model(train_loader_d1, val_loader_d1, model, n_epochs, lr, device, pos_weight=pos_weight)


        elif (loss_function == "FocalLoss"):

            alpha_d1 = helpers.get_alpha_FLoss(train_d1, target_label + ' Label')
            print(f"Focal Loss with alpha = {alpha_d1} ---- dataset {train_dataset}")


            model = train_utils.train_model_with_focal_loss(train_loader_d1, val_loader_d1, model, n_epochs, lr, device, alpha=alpha_d1, gamma=2)
            
        else:
            print("ERR: Loss function must be CrossEntropyLoss or FocalLoss.")

    elif(train_dataset == "MBOD 2"):
        print("Training on Dataset 2\n")

        if(loss_function == "CrossEntropyLoss"):

            if(oversampling):
                pos_weight = helpers.compute_pos_weight(train_d2, target_label + ' Label')

                print(f"Oversampling with pos_weight = {pos_weight} ---- dataset {train_dataset}")

            else:
                pos_weight = torch.tensor([1.0])

            model = train_utils.train_model(train_loader_d2, val_loader_d2, model, n_epochs, lr, device, pos_weight=pos_weight)
            


        elif(loss_function == "FocalLoss"):
            alpha_d2 = helpers.get_alpha_FLoss(train_d2, target_label + ' Label')
            print(f"Focal Loss with alpha = {alpha_d2} ---- dataset {train_dataset}")


            model = train_utils.train_model_with_focal_loss(train_loader_d2, val_loader_d2, model, n_epochs, lr, device, alpha=alpha_d2, gamma=2)
    else:
        print("ERR: Unrecognized dataset name.")

    test_labels_d1, test_preds_d1 = train_utils.test_model(test_loader_d1, model, device, "MBOD 1")
    test_labels_d2, test_preds_d2 = train_utils.test_model(test_loader_d2, model, device, "MBOD 2")

    # Generate and save confusion matrices
    report_d1 = classification_report(test_labels_d1, test_preds_d1)
    cm_d1 = confusion_matrix(test_labels_d1, test_preds_d1)

    report_d2 = classification_report(test_labels_d2, test_preds_d2)
    cm_d2 = confusion_matrix(test_labels_d2, test_preds_d2)

    print(f"Classification Report ({experiment_name}- MBOD 1):")
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

    plt.savefig(f"v4_plots/conf_mat_{experiment_name}.png")
    wandb.log({f"cm-{experiment_name}": wandb.Image(plt)})

    wandb.finish()