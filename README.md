```markdown
# Fashion MNIST CNN Classifier — TensorFlow & Keras

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br/>

**A deep Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify 10 fashion categories from the Fashion-MNIST dataset, achieving 93.89% test accuracy.**

<br/>

</div>

<br>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [CNN Theory](#-cnn-theory)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Inference](#-inference)
- [Demo](#-demo)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Limitations](#-limitations)

<br>

---

## 🔍 Project Overview

This project demonstrates the complete lifecycle of a production-grade image classification pipeline:

<div align="center">

| Phase | Description |
|-------|-------------|
| **Theory** | In-depth coverage of CNN fundamentals — convolutions, pooling, activation functions, backpropagation, and famous architectures |
| **Data** | Loading, preprocessing, normalization, and augmentation of the Fashion-MNIST dataset |
| **Modelling** | Building a deep 3-block CNN with Batch Normalization, Dropout, and fully connected classification layers |
| **Training** | Training with smart callbacks: `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` |
| **Evaluation** | Full classification report, confusion matrix, and training curve visualization |
| **Inference** | Real-world image upload, preprocessing pipeline, and confidence-bar prediction visualization |

</div>

<br>

---

## 🗃️ Dataset

**Fashion-MNIST** is a benchmark dataset created by Zalando Research as a more challenging replacement for the classic MNIST digit dataset.

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*fQiUzijdHlukruf9akgH2Q.png" width="700" alt="Fashion-MNIST Dataset Preview"/>
</div>

<br/>

### Dataset Structure

<div align="center">

| Split | Images | Class Balance |
|-------|--------|---------------|
| Training | 60,000 | 6,000 per class |
| Test | 10,000 | 1,000 per class |
| **Total** | **70,000** | **Perfectly balanced** |

</div>

### Categories & Labels

<div align="center">

| Label | Category | Label | Category |
|-------|----------|-------|----------|
| 0 | 👕 T-shirt/top | 5 | 👡 Sandal |
| 1 | 👖 Trouser | 6 | 👔 Shirt |
| 2 | 🧥 Pullover | 7 | 👟 Sneaker |
| 3 | 👗 Dress | 8 | 👜 Bag |
| 4 | 🧣 Coat | 9 | 👢 Ankle boot |

</div>

### Image Characteristics

<div align="center">

| Property | Value |
|----------|-------|
| Dimensions | 28 × 28 pixels |
| Color Space | Grayscale (single channel) |
| Pixel Value Range | 0 – 255 (normalized to 0 – 1) |
| Total Dataset Size | ~30 MB compressed |

</div>

<br>

---

## 🧠 CNN Theory

> This project includes a comprehensive theoretical introduction to Convolutional Neural Networks.

<details>
<summary><strong>1. The Problem with MLPs for Image Data</strong></summary>

Standard Multi-Layer Perceptrons suffer from two critical failures when applied to images:

- **Parameter Explosion:** A 100×100×3 image requires 30,000 weights just for the first layer neuron.
- **Loss of Spatial Information:** Flattening an image into a 1D vector destroys all spatial and structural relationships between pixels.

</details>

<details>
<summary><strong>2. How CNNs Solve This</strong></summary>

CNNs address these limitations through three core principles:

- **Local Receptive Fields** — Each neuron connects only to a small spatial region of the input, dramatically reducing parameters.
- **Weight Sharing** — The same learned kernel is applied across the entire input, encoding translation invariance.
- **Hierarchical Feature Learning** — Early layers detect edges and textures; deeper layers combine these into complex semantic representations.

</details>

<details>
<summary><strong>3. Core Building Blocks</strong></summary>

<div align="center">

| Layer | Role |
|-------|------|
| **Convolutional Layer** | Slides a learnable kernel across the input to produce a feature map |
| **ReLU Activation** | Introduces non-linearity: `f(x) = max(0, x)` |
| **Pooling Layer** | Downsamples feature maps to reduce computation and improve translation invariance |
| **Batch Normalization** | Normalizes activations per mini-batch to stabilize and accelerate training |
| **Dropout** | Randomly deactivates neurons during training to combat overfitting |
| **Fully Connected Layers** | Combines extracted features for final classification |
| **Softmax Output** | Converts raw logits into a valid probability distribution over 10 classes |

</div>

</details>

<details>
<summary><strong>4. The Training Process</strong></summary>

1. **Forward Pass** → Input flows through the network to produce a prediction.
2. **Loss Function** → `Categorical Cross-Entropy` measures prediction error.
3. **Backpropagation** → Gradients of the loss are computed with respect to all weights.
4. **Optimizer (Adam)** → Updates weights using adaptive learning rates to minimize the loss.

</details>

<details>
<summary><strong>5. Notable Architectures Referenced</strong></summary>

<div align="center">

| Architecture | Year | Key Innovation |
|---|---|---|
| LeNet-5 | 1998 | Pioneering CNN for digit recognition |
| AlexNet | 2012 | Deep CNNs + ReLU + Dropout at scale |
| VGGNet | 2014 | Depth through stacked 3×3 convolutions |
| GoogLeNet | 2014 | Inception modules with parallel convolutions |
| ResNet | 2015 | Residual skip connections for very deep networks |

</div>

</details>

<br>

---

## 🏗️ Model Architecture

The model follows a classic deep CNN pattern with three convolutional blocks followed by a fully connected classifier head:

```
INPUT (28×28×1)
    │
    ▼
┌─────────────────────────────────┐
│  BLOCK 1 — Conv2D(32) × 2      │
│  BatchNorm → MaxPool → Dropout  │  Output: 14×14×32
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  BLOCK 2 — Conv2D(64) × 2      │
│  BatchNorm → MaxPool → Dropout  │  Output: 7×7×64
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  BLOCK 3 — Conv2D(128) × 2     │
│  BatchNorm → MaxPool → Dropout  │  Output: 3×3×128
└─────────────────────────────────┘
    │
    ▼
 FLATTEN → Dense(512) → BN → Dropout(0.5)
    │
    ▼
 Dense(256) → BN → Dropout(0.5)
    │
    ▼
 Dense(10, Softmax) ← OUTPUT
```

### Model Parameters

<div align="center">

| Metric | Value |
|--------|-------|
| Total Parameters | 1,015,530 |
| Trainable Parameters | 1,013,098 |
| Non-Trainable Parameters | 2,432 |
| Model Size | ~3.87 MB |

</div>

<br>

---

## 🔧 Training Pipeline

### Data Augmentation

Real-time augmentation was applied during training using `ImageDataGenerator`:

```python
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### Callbacks

<div align="center">

| Callback | Configuration | Purpose |
|----------|--------------|---------|
| `ModelCheckpoint` | `monitor='val_accuracy'` | Saves the best model weights to Google Drive |
| `EarlyStopping` | `patience=15` | Halts training when validation accuracy plateaus |
| `ReduceLROnPlateau` | `factor=0.5, patience=5` | Halves the learning rate upon loss stagnation |

</div>

### Hyperparameters

<div align="center">

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 128 |
| Max Epochs | 100 |
| Early Stopping Epoch | 67 (best weights from Epoch 52) |
| Random Seed | 42 |

</div>

<br>

---

## 📊 Results

### Test Set Performance

<div align="center">

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **93.89%** |
| **Test Loss** | **0.1721** |

</div>

### Classification Report

<div align="center">

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| T-shirt/top | 0.90 | 0.87 | 0.89 |
| Trouser | **1.00** | **1.00** | **1.00** |
| Pullover | 0.93 | 0.91 | 0.92 |
| Dress | 0.95 | 0.93 | 0.94 |
| Coat | 0.93 | 0.90 | 0.91 |
| Sandal | 0.98 | 0.99 | 0.99 |
| Shirt | 0.78 | 0.85 | 0.81 |
| Sneaker | 0.97 | 0.97 | 0.97 |
| Bag | 0.99 | **1.00** | 0.99 |
| Ankle boot | 0.99 | 0.96 | 0.97 |
| **Macro Avg** | **0.94** | **0.94** | **0.94** |

</div>

> **Key Insight:** The `Shirt` class achieves the lowest F1-score (0.81) due to visual similarity with `T-shirt/top` and `Coat` — a well-documented challenge in the Fashion-MNIST literature. `Trouser` achieves a perfect 1.00 F1-score.

### Training Progression (Selected Epochs)

<div align="center">

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1 | 59.02% | 40.56% |
| 7 | 87.14% | 89.45% |
| 16 | 90.00% | 92.00% |
| 32 | 92.29% | 93.22% |
| **52 (Best)** | **93.75%** | **93.89%** |
| 67 (Stopped) | 94.03% | — |

</div>

<br>

---

## 🔬 Inference

The notebook includes a complete real-world inference pipeline for classifying custom uploaded images.

### Preprocessing Pipeline

Any uploaded image is automatically transformed to match the Fashion-MNIST format:

```
User Image (any size, any format)
        │
        ▼
  Convert to Grayscale
        │
        ▼
  Resize to 28×28 (LANCZOS)
        │
        ▼
  Invert Pixels (white background → black)
        │
        ▼
  Normalize to [0, 1]
        │
        ▼
  Reshape to (1, 28, 28, 1)
        │
        ▼
     Model Input
```

### Sample Inference Output

```
📸 Analyzing: download.jpg
✅ Final Prediction: Shirt (68.74% confidence)
```

The output visualization includes three panels:
1. **Original uploaded image**
2. **Preprocessed 28×28 grayscale image** (model input)
3. **Horizontal confidence bar chart** for all 10 classes (sorted by confidence)

<br>

---

## 🎮 Demo

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cnn-fashion-mnist-image-classifier.streamlit.app/)

<br/>

<img src="https://github.com/MarpakaPradeepSai/CNN-Fashion-MNIST-Image-Classifier/blob/main/MNIST.gif?raw=true" width="700" alt="App Demo"/>

</div>

<br>

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Connect to a GPU runtime: `Runtime → Change runtime type → T4 GPU`
3. Mount Google Drive when prompted
4. Run all cells sequentially (`Runtime → Run all`)

### Option 2: Local Setup

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/MarpakaPradeepSai/CNN-Fashion-MNIST-Image-Classifier.git
cd CNN-Fashion-MNIST-Image-Classifier

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Notebook/Fashion_MNIST_TF.ipynb
```

> **Note:** Remove or comment out all `from google.colab import drive` and `drive.mount(...)` calls when running locally. Update `model_path` to a local directory of your choice.

<br>

---

## 📁 Project Structure

```
CNN-Fashion-MNIST-Image-Classifier/
│
├── 📁 Model/                            # Saved model directory
│   ├── 📄 README.md
│   └── 🧠 fashion_mnist_best_model.keras
│
├── 📁 Notebook/                         # Training notebook directory
│   ├── 📓 Fashion_MNIST_TF.ipynb        # Main notebook (theory + training + inference)
│   └── 📄 README.md
│
├── 🐍 fashion.py                        # Streamlit app script
├── 📄 requirements.txt                  # Python dependencies
├── 📄 README.md                         # Project documentation
└── 📄 LICENSE
```

<br>

---

## 📦 Dependencies

```txt
tensorflow>=2.10.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
seaborn>=0.12.0
Pillow>=9.0.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

<br>

---

## ⚠️ Limitations

<div align="center">

| Limitation | Detail |
|------------|--------|
| **Class Confusion** | Shirts, T-shirts, and Coats share overlapping visual features, resulting in the lowest per-class F1-score |
| **Low Resolution** | The 28×28 pixel constraint limits the fine-grained detail available to the model |
| **Grayscale Only** | Color information — often critical for fashion classification — is absent |
| **Domain Gap** | Real-world images require careful preprocessing to match the dataset's centered, dark-background format |
| **No Rotational Invariance** | Standard CNNs do not inherently handle rotated inputs |

</div>

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

<br>

---

## 🙏 Acknowledgements

- **[Zalando Research](https://github.com/zalandoresearch/fashion-mnist)** — for creating and open-sourcing the Fashion-MNIST dataset
- **[TensorFlow / Keras Team](https://www.tensorflow.org/)** — for the deep learning framework
- **[Yann LeCun et al.](http://yann.lecun.com/exdb/lenet/)** — for the pioneering work on CNNs

<br>

---

<div align="center">

Made with ❤️ and a lot of convolutions

⭐ **Star this repo if you found it helpful!** ⭐

</div>
```
