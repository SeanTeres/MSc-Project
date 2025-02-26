import wandb
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import helpers
import train_utils
import classes

def log_augmented_images(dataset, num_images=5):
    images = []
    for i in range(num_images):
        img, label = dataset[i]
        images.append(wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"Label: {label}"))
    wandb.log({"Augmented Images": images})

# Load the configuration file
with open('classification/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Define model-resolution pairs
model_configs = [
    {
        'model': 'densenet121-res224-all',
        'resolution': 224
    },
    {
        'model': 'resnet50-res512-all',
        'resolution': 512
    }
]

# Update sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'distribution': 'uniform',
            'min': 0.0009,
            'max': 0.002
        },
        'n_epochs': {
            'distribution': 'categorical',
            'values': [30]
        },
        'batch_size': {
            'distribution': 'categorical',
            'values': [32, 48, 64]
        },
        'train_dataset': {
            'distribution': 'categorical',
            'values': ['MBOD 1', 'MBOD 2']
        },
        'augmentation': {
            'distribution': 'categorical',
            'values': [False, True]
        },
        'model_resolution': {
            'distribution': 'categorical',
            'values': [224]
        },
        'oversampling': {
            'distribution': 'categorical',
            'values': [True]
        },
        'classifier': {
            'distribution': 'categorical',
            'values': ['Base']
        },
        'loss_function': {
            'distribution': 'categorical',
            'values': ['FocalLoss']
        },
        'alphas_FLoss': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }, 
        'target': {
            'distribution': 'categorical',
            'values': ['Profusion', 'TBA-TBU']
        }
    }
}


def train():
    # Initialize wandb run
    run = wandb.init()
    
    # Access all hyperparameters through wandb.config
    config = wandb.config
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datasets
    dicom_dir_1 = 'MBOD_Datasets/Dataset 1'
    dicom_dir_2 = 'MBOD_Datasets/Dataset 2'
    metadata_1 = pd.read_excel('MBOD_Datasets/Dataset 1/FileDatabaseWithRadiology.xlsx')
    metadata_2 = pd.read_excel('MBOD_Datasets/Dataset 2/Database_Training-2024.08.28.xlsx')

    # Extract model configuration
    # model_name = config.model_config['model']
    model_resolution = config.model_resolution
    
    
    # Initialize model based on config
    if model_resolution == 224:
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.classifier = classes.BaseClassifier(in_features=1024).to(device)
    else:
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model.features2 = model.features
        model.classifier = classes.BaseClassifier512(in_features=2048).to(device)

    model = model.to(device)

    # model_config = config.model_config


    # Initialize datasets and dataloaders based on config
    d1 = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
    d2 = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2)

    train_indices_d1, val_indices_d1, test_indices_d1 = helpers.split_dataset(d1)
    train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2)

    # Save indices for later use
    split_indices = {
        'd1': {'train': train_indices_d1, 'val': val_indices_d1, 'test': test_indices_d1},
        'd2': {'train': train_indices_d2, 'val': val_indices_d2, 'test': test_indices_d2}
    }
        
    d1.set_target(config.target, model_resolution)
    d2.set_target(config.target, model_resolution)

        # Create the transformation pipeline
    augmentations_list = transforms.RandomApply([
        # transforms.CenterCrop(np.round(224 * 0.9).astype(int)),  # Example crop
        transforms.RandomRotation(degrees=(-5, 5)),  
        transforms.Lambda(lambda img: helpers.salt_and_pepper_noise_tensor(img, prob=0.02)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.025, 0.025))
    ], p=0.5) 

    # Create augmented datasets
    d1_aug = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1, transform=augmentations_list)
    d2_aug = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2, transform=augmentations_list)

    d1_aug.set_target(config.target, model_resolution)
    d2_aug.set_target(config.target, model_resolution)
    

    train_d1 = Subset(d1, train_indices_d1)
    train_aug_d1 = Subset(d1_aug, train_indices_d1)

    val_d1 = Subset(d1, val_indices_d1)
    test_d1 = Subset(d1, test_indices_d1)

    train_d2 = Subset(d2, train_indices_d2)
    train_aug_d2 = Subset(d2_aug, train_indices_d2)

    val_d2 = Subset(d2, val_indices_d2)
    test_d2 = Subset(d2, test_indices_d2)

    print(f'Length of D1: {len(train_d1)}')
    print(f'Length of D2: {len(train_d2)}')

    # Create dataloaders
    if config.train_dataset == "MBOD 1":
        train_loader, aug_train_loader, val_loader, _ = helpers.create_dataloaders(
            train_d1, train_aug_d1, val_d1, None, config.batch_size, target=config.target
        )
        
    elif config.train_dataset == "MBOD 2":
        train_loader, aug_train_loader, val_loader, _ = helpers.create_dataloaders(
            train_d2, train_aug_d2, val_d2, None, config.batch_size, target=config.target
        )
    
    if config.augmentation:
        train_loader = aug_train_loader
        log_augmented_images(train_aug_d1 if config.train_dataset == "MBOD 1" else train_aug_d2, num_images=5)
    else:
        train_loader = train_loader
    
    # Training loop
    if config.loss_function == "CrossEntropyLoss":
        if config.oversampling:
            pos_weight = helpers.compute_pos_weight(
                train_d1 if config.train_dataset == "MBOD 1" else train_d2, 
                config.target + ' Label'
            )
        else:
            pos_weight = torch.tensor([1.0])
            
        model = train_utils.train_model(
            train_loader, val_loader, model, config.n_epochs, 
            config.lr, device, pos_weight=pos_weight, 
            experiment_name=wandb.run.name
        )
    
    elif config.loss_function == "FocalLoss":
        alpha = helpers.get_alpha_FLoss(
            train_d1 if config.train_dataset == "MBOD 1" else train_d2,
            config.target + ' Label'
        )

        rand_alpha = config.alphas_FLoss
        print(f"ORIGINAL: {alpha}")
        # print(f"RANDOM: {rand_alpha}")
        # alpha = min(alpha, rand_alpha)
        model = train_utils.train_model_with_focal_loss(
            train_loader, val_loader, model, config.n_epochs,
            config.lr, device, alpha=rand_alpha, gamma=2,
            experiment_name=wandb.run.name
        )

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='MBOD-SWEEP')

# Start the sweep
wandb.agent(sweep_id, function=train, count=15)