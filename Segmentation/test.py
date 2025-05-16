import numpy as np
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from data_set import FireDataset
from PIL import Image,ImageDraw
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from model.segformer import Segformer
from model.unetpp import UnetPlusPlus
from model.abcnet import ABCNet
from model.unet import UNet
from model.TransUNet.networks.vit_seg_modeling import VisionTransformer
import model.TransUNet.networks.vit_seg_configs as configs

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Segformer().to(device)
# model = UnetPlusPlus(2).to(device)
# model = ABCNet().to(device)
model = UNet().to(device)
# config_vit = configs.get_r50_b16_config()
# model = VisionTransformer(config_vit, img_size=256, num_classes=1).to(device)
model.load_state_dict(torch.load('model.pth'))


test_dataset = FireDataset('SatFire Dataset/test/data',
                            'SatFire Dataset/test/label',
                            transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = 'cuda'

total_accuracy = 0
total = 0
total_f1 = 0
total_rec = 0
total_pre = 0
total_miou = 0
total_fwiou = 0
total_kappa = 0
num = 1
with torch.no_grad():
    for images, masks in test_dataloader:
        images = images.to(device)
        # masks = masks.to(device)

        outputs = model(images)

        outputs = outputs[-1]

        y = torch.where(outputs > 0.5, torch.tensor(1.0),
                        torch.tensor(0.0))
        y = y.cpu().numpy()
        y_pred = y.squeeze()

        width, height = 256, 256
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        # Iterate through the array and fill the cubes according to the values of the array elements
        for hh in range(height):
            for ww in range(width):
                fill_color = 'black' if y_pred[hh][ww] == 0.0 else 'white'
                draw.rectangle([ww, hh, ww + 1, hh + 1], fill=fill_color)


        img.save('image_out/SatFireNet/'+str(num)+'.png')
        num += 1

        masks = masks.cpu().numpy().squeeze()
        correct = np.sum(y == masks)

        size_fig = 256 * 256

        y = y_pred

        average_pixel_accuracy = correct / size_fig
        total += average_pixel_accuracy
        unique_classes = np.unique(masks)
        ious = []
        class_counts = []
        for cls in unique_classes:
            pred_cls = (y == cls).astype(int)
            label_cls = (masks == cls).astype(int)
            intersection = np.sum(pred_cls * label_cls)
            union = np.sum(pred_cls + label_cls - pred_cls * label_cls)
            iou = intersection / union if union != 0 else 1
            ious.append(iou)
            class_counts.append(np.sum(label_cls))
        mean_iou = np.mean(ious)
        total_miou += mean_iou

        class_weights = np.array(class_counts, dtype=float)
        class_weights /= np.sum(class_weights)  # normalized weight
        fw_iou = np.sum(np.array(ious) * class_weights)
        total_fwiou += fw_iou

        pred_flat = y_pred.flatten() > 0
        label_flat = masks.flatten() > 0

        # Calculate the confusion matrix
        cm = confusion_matrix(label_flat, pred_flat)

        # kappa
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        # Calculation of observational consistency (Po)
        Po = (TP + TN) / (TP + TN + FP + FN)

        # Calculate the desired consistency (Pe)
        Pe = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / ((TP + TN + FP + FN) ** 2)

        # Calculation of the Kappa coefficient
        kappa = (Po - Pe) / (1 - Pe) if Pe != 1 else 1.0
        total_kappa += kappa

        # Calculating Precision, Recall, and F1 Scores
        precision = precision_score(label_flat, pred_flat)
        total_pre += precision

        recall = recall_score(label_flat, pred_flat)
        total_rec += recall

        f1 = f1_score(label_flat, pred_flat)
        total_f1 += f1

average_accuracy = total / len(test_dataloader)
print(f'Average accuracy on test set: {average_accuracy:.4f}')

average_miou = total_miou / len(test_dataloader)
print(f'Average miou on test set: {average_miou:.4f}')

average_fwiou = total_fwiou / len(test_dataloader)
print(f'Average fwiou on test set: {average_fwiou:.4f}')

average_pre = total_pre / len(test_dataloader)
print(f'Average pre on test set: {average_pre:.4f}')

average_rec = total_rec / len(test_dataloader)
print(f'Average rec on test set: {average_rec:.4f}')

average_f1 = total_f1 / len(test_dataloader)
print(f'Average f1 on test set: {average_f1:.4f}')

average_kappa = total_kappa / len(test_dataloader)
print(f'Average kappa on test set: {average_kappa:.4f}')

