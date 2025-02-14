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
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time
import random


from helpers import read_and_normalize_xray, split_with_indices, create_dataloaders
from classes import DICOMDataset1, DICOMDataset2, AugmentedDataset, BaseClassifier
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

def train_model(train_loader, val_loader, model, n_epochs, lr, device, pos_weight):
    """Function to train a model on a given training dataloader."""
    
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))  # Use BCEWithLogitsLoss for binary classification
    optim_1 = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  # Start timing

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train()  # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            print(f"Batch: {idx+1}/{len(train_loader)}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            optim_1.zero_grad()

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            corr += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_acc = corr / tot

        # Log metrics every batch

            wandb.log({
                "batch": (idx + 1),
                "batch_loss": loss.item(),
                "batch_accuracy": batch_acc
            })

            del imgs, labels, output, features
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1
        })

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

                features = model.features2(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_preds, average='weighted')
        val_epoch_recall = recall_score(val_labels, val_preds, average='weighted')
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')

        val_kappa = cohen_kappa_score(val_labels, val_preds)

        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_precision": val_epoch_precision,
            "val_recall": val_epoch_recall,
            "val_f1": val_epoch_f1,
            "val_kappa": val_kappa
        })

        # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}, "
              f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}, "
              f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, Validation F1: {val_epoch_f1:.4f}, Validation Kappa: {val_kappa}")

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return model


def train_model_with_focal_loss(train_loader, val_loader, model, n_epochs, lr, device, alpha, gamma):
    """Function to train a model on a given training dataloader using Binary Focal Loss."""
    
    model = model.to(device)

    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    optim_1 = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  # Start timing

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train()  # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            print(f"Batch: {idx+1}/{len(train_loader)}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            optim_1.zero_grad()

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            corr += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_acc = corr / tot

        # Log metrics every batch

            wandb.log({
                "batch": idx + 1,
                "batch_loss": loss.item(),
                "batch_accuracy": batch_acc
            })

            del imgs, labels, output, features
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1
        })

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

                features = model.features2(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_preds, average='weighted')
        val_epoch_recall = recall_score(val_labels, val_preds, average='weighted')
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_kappa = cohen_kappa_score(val_labels, val_preds)

        print(f"Labels: {val_labels}")
        print(f"Predictions: {val_preds}")

        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_precision": val_epoch_precision,
            "val_recall": val_epoch_recall,
            "val_f1": val_epoch_f1,
            "val_kappa": val_kappa
        })

        # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}, "
              f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}, "
              f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, Validation F1: {val_epoch_f1:.4f}, "
              f"Validation Kappa: {val_kappa}")

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return model


def test_model(test_loader, model, device, test_dataset_name):
    """Function to evaluate a trained model on a specific test loader.
    Returns the true labels and predicted labels for further analysis."""
    criterion = nn.BCEWithLogitsLoss()

    label_mapping = {0: "None", 1: "Profusion ≥ 1/1 and TBA/TBU"}

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            test_running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

            del imgs, labels, output, features
            torch.cuda.empty_cache()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_kappa = cohen_kappa_score(test_labels, test_preds)

    # Log test metrics
    wandb.log({
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_kappa": test_kappa
    })

    # Print test results
    print(f"Test Results for {test_dataset_name} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}, "
        f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, "
        f"Cohen's Kappa: {test_kappa:.4f}")

    print("****"*25)

    return test_labels, test_preds

def train_model_aug(train_loader, val_loader, model, n_epochs, lr, device, p_aug):
    """Function to train a model on a given training dataloader."""
    # Define augmentations
    augmentations_list = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ]
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optim_1 = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  # Start timing

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train()  # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            print(f"Batch: {idx+1}/{len(train_loader)}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            random_number = random.random()
            print(f"Random value: {random_number}")

            if random_number >= p_aug:
                print("Augmenting batch...")

                imgs = torch.stack([augmentations_list(img) for img in imgs])

            optim_1.zero_grad()

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            total += labels.size(0) 
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            corr += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_acc = corr / tot

        # Log metrics every batch

            wandb.log({
                "batch": idx + 1,
                "batch_loss": loss.item(),
                "batch_accuracy": batch_acc
            })

            del imgs, labels, output, features
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1
        })

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

                features = model.features2(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_preds, average='weighted')
        val_epoch_recall = recall_score(val_labels, val_preds, average='weighted')
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')

        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_precision": val_epoch_precision,
            "val_recall": val_epoch_recall,
            "val_f1": val_epoch_f1
        })

        # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}, "
              f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}, "
              f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, Validation F1: {val_epoch_f1:.4f}")

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return model
