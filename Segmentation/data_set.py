import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


class FireDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])

        index_gt = self.images[idx].find('.')
        mask_sub = self.images[idx][0:index_gt] + '_gt' + self.images[idx][index_gt :]

        # mask_name = os.path.join(self.masks_dir, mask_sub)
        mask_name = os.path.join(self.masks_dir, mask_sub)

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        # if self.transform:
        image = self.transform(image)
        mask = self.transform(mask)


        return image, mask

