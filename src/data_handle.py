import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip,\
      RandomRotation, RandomPhotometricDistort, ToDtype, Normalize, Resize, ToImage
from torchvision.datasets import ImageFolder

def process_data(config):
    train_transforms = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.2),
        RandomVerticalFlip(p=0.2),
        RandomRotation(60, fill=200),
        RandomPhotometricDistort(p=0.2, contrast=[0.8, 1.2], hue=[-0.1, 0.1]),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = Compose([
        Resize((224, 224)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(os.path.join(config["data"]["data_dir"], "train"), transform=train_transforms)
    val_dataset = ImageFolder(os.path.join(config["data"]["data_dir"], "test"), transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["train_batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["data"]["val_batch_size"],
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )     
    
    return train_loader, val_loader