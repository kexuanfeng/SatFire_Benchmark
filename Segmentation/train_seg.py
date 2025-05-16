import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from data_set import FireDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
from model.segformer import Segformer
from model.unetpp import UnetPlusPlus
from model.abcnet import ABCNet
from model.unet import UNet
from model.TransUNet.networks.vit_seg_modeling import VisionTransformer
import model.TransUNet.networks.vit_seg_configs as configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# model = Segformer().to(device)
# model = UnetPlusPlus(2).to(device)
# model = ABCNet().to(device)
model = UNet().to(device)
# config_vit = configs.get_r50_b16_config()
# model = VisionTransformer(config_vit, img_size=256, num_classes=1).to(device)


transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

train_dataset = FireDataset('SatFire Dataset/train/data',
                            'SatFire Dataset/train/label',
                            transform=transform)


test_dataset = FireDataset('SatFire Dataset/test/data',
                            'SatFire Dataset/test/label',
                            transform=transform)


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

transform = transforms.Compose([
    transforms.ToTensor(),
])




class BCEWithLogits(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, **kwargs):
        super(BCEWithLogits, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, pred, target):
        loss = self.crit(pred, target)
        return loss

# criterion = BCEWithLogits()
criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
best_miou = 0

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    print('start train')
    for images, masks in train_dataloader:
        images = images.to(device)
        masks = masks.to(device)
        # print(masks,masks.dtype)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs[-1]

        loss = criterion(outputs.view(1,1,256,256), masks)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    total_accuracy = 0
    total = 0
    ioutttt = 0
    total_f1 = 0
    total_rec = 0
    total_pre = 0
    total_miou = 0
    total_fwiou = 0
    total_kappa = 0
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

            # print(y.shape)
            masks = masks.cpu().numpy().squeeze()
            # print(masks)
            correct = np.sum(y == masks)
            sizehh = 256 * 256

            y = y_pred

            average_pixel_accuracy = correct / sizehh
            # accuracy = calculate_accuracy(y, masks)
            total += average_pixel_accuracy
            # print(f'Current batch accuracy: {average_pixel_accuracy:.4f}')

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
            # print(mean_iou)
            total_miou += mean_iou

            class_weights = np.array(class_counts, dtype=float)
            class_weights /= np.sum(class_weights)
            fw_iou = np.sum(np.array(ious) * class_weights)
            total_fwiou += fw_iou

            pred_flat = y_pred.flatten() > 0
            label_flat = masks.flatten() > 0

            cm = confusion_matrix(label_flat, pred_flat)

            #kappa
            TP = cm[1, 1]
            TN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]

            # Po
            Po = (TP + TN) / (TP + TN + FP + FN)

            # Pe
            Pe = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / ((TP + TN + FP + FN) ** 2)

            kappa = (Po - Pe) / (1 - Pe) if Pe != 1 else 1.0

            total_kappa += kappa

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

    if average_miou > best_miou:
        torch.save(model.state_dict(), r'.\results\model.pth')
        best_miou = average_miou
        best_fwiou = average_fwiou
        best_pre = average_pre
        best_rec = average_rec
        best_f1 = average_f1
        best_kappa = average_kappa
