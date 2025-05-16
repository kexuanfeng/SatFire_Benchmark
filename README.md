# SatFire Benchmark

**SatFire Benchmark** provides standardized code for classification and segmentation tasks based on the [SatFire dataset](https://huggingface.co/datasets/kexuan1021/SatFire), which focuses on active wildfire detection and analysis from satellite remote sensing imagery.

## 📂 Dataset
![Example Image](SatFire.jpg)  

The SatFire dataset includes high-resolution optical remote sensing images annotated with fire and non-fire categories, as well as pixel-level masks for active fire segmentation.

- 🔗 Hugging Face: [https://huggingface.co/datasets/kexuan1021/SatFire](https://huggingface.co/datasets/kexuan1021/SatFire)
- Categories: `fire`, `non-fire`
- Tasks: Image classification, semantic segmentation

## 📦 Repository Structure

```
SatFire_Benchmark/
├── classification/        # Classification training & evaluation code
│   ├── satfire_classification_tsne.py
├── segmentation/          # Segmentation training & evaluation code
│   ├── train_seg.py
│   ├── data_set.py
│   └── test.py
└── README.md
```

## 🚀 Quick Start

### 1. Download the dataset

Download the dataset manually from [Hugging Face](https://huggingface.co/datasets/kexuan1021/SatFire) and place it under a suitable directory (e.g., `./data/SatFire`).

### 2. Run classification

```bash
cd classification
python train.py
```

### 3. Run segmentation

```bash
cd segmentation
python train_seg.py 
```

## 🧪 Tasks

### 🔤 Classification

- Input: satellite images
- Output: Binary label (`fire` or `non-fire`)
- Metrics: Accuracy, Precision, Recall, F1-score

### 🎯 Segmentation

- Input: RGB satellite images
- Output: Pixel-wise fire segmentation masks
- Metrics: mIoU, Dice coefficient

## 📮 Contact

For questions or contributions, feel free to open an issue or contact:

- GitHub: [@kexuanfeng](https://github.com/kexuanfeng)

---

🔥 Let's work together to build stronger wildfire detection systems using AI and satellite data!
