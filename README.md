
# 🔥 SatFire Benchmark: Wildfire Detection and Segmentation

本项目包含基于 [SatFire 数据集](https://huggingface.co/datasets/kexuan1021/SatFire) 的分类与分割实验代码，旨在推动遥感图像中野火检测和分割任务的研究。

## 📁 项目结构

```
SatFire_Benchmark/
├── Classification/      # 野火图像分类任务
│   ├── train.py         # 分类模型训练脚本
│   ├── test.py          # 分类模型测试脚本
│   └── model.py         # 模型定义
├── Segmentation/        # 野火图像分割任务
│   ├── train.py         # 分割模型训练脚本
│   ├── test.py          # 分割模型测试脚本
│   └── unet.py          # UNet 模型定义
├── dataset/             # 数据加载器
│   └── satfire_loader.py
├── utils/               # 通用工具函数
│   └── transforms.py
├── requirements.txt     # Python 依赖包列表
└── README.md            # 项目说明文件
```

## 📦 环境安装

推荐使用 Conda 创建隔离环境：

```bash
conda create -n satfire python=3.10 -y
conda activate satfire
pip install -r requirements.txt
```

## 🔍 数据集说明

本项目使用 [SatFire 数据集](https://huggingface.co/datasets/kexuan1021/SatFire)，该数据集包含多个遥感图像类别，涵盖正在燃烧的野火区域及其对应的分割标签。

您可通过 Hugging Face 加载数据集：

```python
from datasets import load_dataset
dataset = load_dataset("kexuan1021/SatFire")
```

数据结构包括：
- 图像（image）
- 分类标签（label）
- 分割掩码（mask）

## 🧠 分类任务

训练分类模型（默认使用 ResNet）：

```bash
cd Classification
python train.py --config config.yaml
```

测试分类模型：

```bash
python test.py --weights path/to/model.pth
```

## 🎯 分割任务

训练分割模型（默认使用 UNet）：

```bash
cd Segmentation
python train.py --config config.yaml
```

测试分割模型：

```bash
python test.py --weights path/to/model.pth
```

## 📊 评估指标

- 分类任务：
  - 准确率（Accuracy）
  - 精确率/召回率/F1 值
- 分割任务：
  - 交并比（IoU）
  - Dice 系数

## 📄 许可证

本项目基于 MIT License 开源，详见 [LICENSE](LICENSE)。

## 🙌 鸣谢

- 感谢 Hugging Face Datasets 提供平台支持
- 感谢 PyTorch、Albumentations、Torchvision 等开源工具库

---

📫 如有问题欢迎提 issue 或联系作者：[kexuanfeng](https://github.com/kexuanfeng)
