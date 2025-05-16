import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SatFire dataset
class FireDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                label = 0 if 'no' in filename else 1
                self.image_paths.append(os.path.join(folder_path, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_loaders(data_dir, image_size=224, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_set = FireDataset(os.path.join(data_dir, 'train'), transform)
    test_set = FireDataset(os.path.join(data_dir, 'test'), transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# feature
class FeatureModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        feats = self.base_model.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            feats = feats.mean(dim=[2, 3])  # GAP
        return feats

# load model
def get_model(model_name, num_classes=2, feature_only=False):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    if feature_only:
        if hasattr(model, 'forward_features'):
            return FeatureModel(model).to(device)
        else:
            raise NotImplementedError(f"Feature extraction not supported for {model_name}")
    return model.to(device)

# train
def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss / len(loader):.4f}")

# test
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1

# t-SNE visual
def tsne_visualization(model_name, model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = feats.view(feats.size(0), -1)  # 👈 flatten
            features.append(feats.cpu())
            labels.extend(lbls)
    features = torch.cat(features).numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_feats = tsne.fit_transform(features)
    tsne_feats = (tsne_feats - tsne_feats.min()) / (tsne_feats.max() - tsne_feats.min())

    plt.figure(figsize=(8,6))
    labels = np.array(labels)
    plt.scatter(tsne_feats[labels==0, 0], tsne_feats[labels==0, 1], c='blue', label='No Fire', alpha=0.6)
    plt.scatter(tsne_feats[labels==1, 0], tsne_feats[labels==1, 1], c='red', label='Fire', alpha=0.6)
    plt.title(f"t-SNE Visualization - {model_name}")
    plt.legend()
    plt.savefig(f"tsne_{model_name}.png")
    plt.close()
    print(f"[{model_name}] 🖼 t-SNE visualization saved to tsne_{model_name}.png")
# main
def main():
    data_path = '/satfire/classification'  # /path/your_dataset
    train_loader, test_loader = get_loaders(data_path, image_size=224, batch_size=32)

    model_names = [
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "vit_b_16",
        "convnext_tiny",
        "beit_base_patch16_224",
        "swin_tiny_patch4_window7_224"
    ]

    for model_name in model_names:
        print(f"\n🚀 Training model: {model_name}")
        model = get_model(model_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        train(model, train_loader, optimizer, criterion, epochs=50)
        acc, prec, rec, f1 = evaluate(model, test_loader)
        print(f"[{model_name}] ✅ Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

        print(f"[{model_name}] 🔍 Extracting features for t-SNE...")
        feature_model = get_model(model_name, feature_only=True)
        feature_model.load_state_dict(model.state_dict(), strict=False)
        feature_model.to(device)
        tsne_visualization(model_name, feature_model, test_loader)

if __name__ == "__main__":
    main()
