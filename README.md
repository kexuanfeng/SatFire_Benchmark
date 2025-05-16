# SatFire Benchmark

**SatFire Benchmark** provides standardized code for classification and segmentation tasks based on the [SatFire dataset](https://huggingface.co/datasets/kexuan1021/SatFire), which focuses on active wildfire detection and analysis from satellite remote sensing imagery.

## 📂 Dataset

The SatFire dataset includes high-resolution optical remote sensing images annotated with fire and non-fire categories, as well as pixel-level masks for active fire segmentation.

- 🔗 Hugging Face: [https://huggingface.co/datasets/kexuan1021/SatFire](https://huggingface.co/datasets/kexuan1021/SatFire)
- Categories: `fire`, `non-fire`
- Tasks: Image classification, semantic segmentation

## 📦 Repository Structure

```
SatFire_Benchmark/
├── classification/        # Classification training & evaluation code
│   ├── train.py
│   ├── dataset.py
│   └── ...
├── segmentation/          # Segmentation training & evaluation code
│   ├── train.py
│   ├── dataset.py
│   └── ...
├── utils/                 # Common tools and helpers
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kexuanfeng/SatFire_Benchmark.git
cd SatFire_Benchmark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the dataset manually from [Hugging Face](https://huggingface.co/datasets/kexuan1021/SatFire) and place it under a suitable directory (e.g., `./data/SatFire`).

### 4. Run classification

```bash
cd classification
python train.py --data_root ../data/SatFire
```

### 5. Run segmentation

```bash
cd segmentation
python train.py --data_root ../data/SatFire
```

## 🧪 Tasks

### 🔤 Classification

- Input: RGB satellite images
- Output: Binary label (`fire` or `non-fire`)
- Metrics: Accuracy, Precision, Recall, F1-score

### 🎯 Segmentation

- Input: RGB satellite images
- Output: Pixel-wise fire segmentation masks
- Metrics: mIoU, Dice coefficient

## 📈 Results

| Task           | Method     | Accuracy / mIoU |
|----------------|------------|------------------|
| Classification | ResNet18   | 95.2%           |
| Segmentation   | UNet       | 81.6% mIoU      |

> Note: These results are based on preliminary experiments. See respective folders for configuration details.

## 📝 Citation

If you use this repository or the SatFire dataset in your work, please cite:

```bibtex
@misc{satfire2025,
  title={SatFire: A Benchmark Dataset for Active Wildfire Detection in Satellite Imagery},
  author={Feng, Kexuan},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/kexuan1021/SatFire}}
}
```

## 📮 Contact

For questions or contributions, feel free to open an issue or contact:

- GitHub: [@kexuanfeng](https://github.com/kexuanfeng)

---

🔥 Let's work together to build stronger wildfire detection systems using AI and satellite data!
