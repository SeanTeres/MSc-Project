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


from helpers import read_and_normalize_xray, split_with_indices, create_dataloaders
from classes import DICOMDataset1, DICOMDataset2, AugmentedDataset, BaseClassifier1

def train_model(train_loader, val_loader, model, n_epochs, lr, device):
    """Function to train a model on a given training dataloader."""

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optim_1 = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train() # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            print(f"Batch: {idx+1}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device)

            optim_1.zero_grad()

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            batch_acc = corr/tot

            del imgs, labels, output, features
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_precision = precision_score(all_labels, all_preds)
        epoch_recall = recall_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds)


        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                features = model.features2(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                _, predicted = torch.max(output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_preds)
        val_epoch_recall = recall_score(val_labels, val_preds)
        val_epoch_f1 = f1_score(val_labels, val_preds)

            # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%, "
            f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%, "
            f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, Validation F1: {val_epoch_f1:.4f}")

                                 
def test_model(test_loader, model, device):
    criterion = nn.CrossEntropyLoss()

    label_mapping = {0: "None", 1: "Profusion ≥ 1/1"}

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            test_running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

            del imgs, labels, output, features
            torch.cuda.empty_cache()

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_acc = test_correct / test_total
    test_epoch_precision = precision_score(test_labels, test_preds)
    test_epoch_recall = recall_score(test_labels, test_preds)
    test_epoch_f1 = f1_score(test_labels, test_preds)
    test_epoch_kappa = cohen_kappa_score(test_labels, test_preds)

    # Print test results
    print(f"Test Results - Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_acc:.2f}, "
        f"Precision: {test_epoch_precision:.4f}, Recall: {test_epoch_recall:.4f}, F1 Score: {test_epoch_f1:.4f}, "
        f"Cohen's Kappa: {test_epoch_kappa:.4f}")

    print("****"*25)

    report = classification_report(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)


    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
                yticklabels=[label_mapping[i] for i in range(len(label_mapping))])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - D1')
    plt.show()



