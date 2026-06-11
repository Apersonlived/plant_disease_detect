# Tomato Disease Detection Using Deep Learning
A comparative study of three CNN architectures for real-time tomato leaf disease classification, featuring standard and real-world evaluation, and a Streamlit deployment interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Project Overview

This project implements and compares three convolutional neural network architectures for tomato leaf disease classification:

1. **Baseline CNN**: a 4-block CNN trained from scratch with no augmentation, serving as the performance reference
2. **Improved CNN**: a deeper 5-block CNN with data augmentation, class-weighted loss, and stronger regularisation
3. **Fine-Tuned ResNet50**: an ImageNet pretrained ResNet50 fine-tuned using a two-phase strategy

Each model is evaluated on two test sets:
- **Standard test set** — drawn from the same PlantVillage/PlantDoc distribution as training data
- **Real-world test set** — PlantDoc field images captured in natural outdoor conditions, never seen during training

This dual evaluation framework reveals the domain gap between laboratory and real-world performance, which is the central finding of the study.

### Disease Classes

| Class | Description |
|---|---|
| Early Blight | Fungal disease caused by *Alternaria solani* — concentric ring lesions on lower leaves |
| Late Blight | Caused by *Phytophthora infestans* — rapid spreading, severe crop damage |
| Healthy | No visible disease symptoms |
| Tomato Yellow Leaf Curl Virus | Viral disease transmitted by whiteflies — no chemical cure |

> **Note:** Leaf Mold was initially included as a fifth class but removed after evaluation due to consistently poor real-world generalisation across all three models and data constraints.

---

## Dataset

### Sources

| Dataset | Source | Description |
|---|---|---|
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | GitHub | 54,305 images across 14 crop species, 26 diseases, 38 classes — laboratory controlled conditions |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | GitHub | 2,598 images across 13 species, 17 disease classes — real-world field images |

PlantVillage images are captured under controlled laboratory conditions with uniform backgrounds and visual consistency while PlantDoc contains diverse real-world images with blurry images, leaves in real environment, and variable lighting (Singh et al., 2020). 

### Combined Dataset Statistics

| Class | Images |
|---|---|
| Tomato Yellow Leaf Curl Virus | 5,580 |
| Late Blight | 2,010 |
| Healthy | 1,650 |
| Early Blight | 1,079 |
| **Total** | **10,319** |

The dataset contains a significant imbalance in class distribution with Yellow Leaf Curl Virus containing almost half the images (5,580 of 10,319).

### Split Strategy

Stratified 70/15/15 train/validation/test split using scikit-learn's `train_test_split` with stratification on class labels to ensure proportional representation of all classes. The PlantDoc test dataset was kept entirely separate for additional real-world evaluation.

```
Train      : ~7,223 images  (augmented for Improved and ResNet models)
Validation : ~1,548 images  (no augmentation)
Test       : ~1,548 images  (no augmentation)
Real-world :     42 images  (PlantDoc test set: separated for further evaluation)
```

---

## Models

### Model 1 — Baseline CNN

A CNN architecture comprising four convolutional blocks with Batch Normalisation, ReLU activation, and Max Pooling applied to each layer (Gurucharan, 2026). The convolution block applies 2D convolution with kernel and padding to preserve spatial dimensions. Each layer uses increased feature maps with each successive block to derive more complex feature maps from low-level features.

```
Input (224×224×3)
├── Block 1: Conv2d(3→32)   → BatchNorm → ReLU → MaxPool  [224→112]
├── Block 2: Conv2d(32→64)  → BatchNorm → ReLU → MaxPool  [112→56]
├── Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool  [56→28]
├── Block 4: Conv2d(128→256)→ BatchNorm → ReLU → MaxPool  [28→14]
├── AdaptiveAvgPool2d(4×4)
└── FC: 4096 → 512 → Dropout(0.4) → 4
```

- **Input size:** 224×224
- **Augmentation:** None — intentional baseline design
- **Loss:** CrossEntropyLoss (class-weighted)
- **Optimiser:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=4, factor=0.5)
- **Epochs:** 20

### Model 2 — Improved CNN

Built upon the baseline architecture using data augmentation and hyperparameter optimisation. Utilises an additional convolution block (256→512) and a double convolution in the 4th block. Also utilises class-weighted cross entropy and regularisation to prevent overfitting, and ReduceLROnPlateau to reduce the learning rate when accuracy does not improve significantly (Al-Kababji et al., 2022).

```
Input (128×128×3)
├── Block 1: Conv2d(3→32)   → BatchNorm → ReLU → MaxPool  [128→64]
├── Block 2: Conv2d(32→64)  → BatchNorm → ReLU → MaxPool  [64→32]
├── Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool  [32→16]
├── Block 4: Conv2d(128→256)→ BatchNorm → ReLU →
│            Conv2d(256→256)→ BatchNorm → ReLU → MaxPool  [16→8]  ← double conv
├── Block 5: Conv2d(256→512)→ BatchNorm → ReLU → MaxPool  [8→4]   ← new block
├── AdaptiveAvgPool2d(2×2)
└── FC: 2048 → 512 → Dropout(0.5) → 4
```

- **Input size:** 128×128 (reduced for computational efficiency)
- **Augmentation:** RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°), ColorJitter, RandomGrayscale, RandomErasing
- **Loss:** CrossEntropyLoss (class-weighted with manual boosts)
- **Optimiser:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Epochs:** 30

### Model 3 — Fine-Tuned ResNet50

Utilises the weights of ResNet50 pretrained on ImageNet, using a residual learning framework with residual connections (He et al., 2016) making it trainable even as the network increases in depth. The 1,000 ImageNet classes are mapped to 4 classes utilising transfer learning. Training is carried out in two phases; Phase 1 freezes all layers and trains only the new classification head, Phase 2 unfreezes later layers for joint fine-tuning.

```
Input (224×224×3)
└── ResNet50 backbone (ImageNet pretrained —
 He et al., 2016)
    ├── Phase 1: All layers frozen → train FC head only (5 epochs,  lr=1e-3)
    └── Phase 2: Unfreeze layer2 + layer3 + layer4 (20 epochs)
                 ├── layer2:  lr=5e-6
                 ├── layer3:  lr=1e-5
                 ├── layer4:  lr=1e-5
                 └── FC head: lr=1e-4
└── FC: 2048 → 4  (replaces original 1000-class head)
```

- **Input size:** 224×224
- **Augmentation:** RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(20°), ColorJitter
- **Loss:** CrossEntropyLoss (class-weighted)
- **Optimiser:** Adam with differential learning rates + weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR

---

## Results

### Standard Test Set

| Model | Accuracy | Macro F1 | Early Blight F1 | Late Blight F1 | Healthy F1 | YLCV F1 |
|---|---|---|---|---|---|---|
| Baseline CNN | 97.74% | 0.9671 | 0.9359 | 0.9544 | 0.9878 | 0.9905 |
| Improved CNN | 96.71% | 0.9494 | 0.8901 | 0.9359 | 0.9819 | 0.9898 |
| Fine-Tuned ResNet50 | **99.03%** | **0.9855** | **0.9726** | **0.9784** | **0.9940** | **0.9970** |

### Real-World Test Set (PlantDoc — out-of-distribution)

| Model | Accuracy | Macro F1 | Early Blight F1 | Late Blight F1 | Healthy F1 | YLCV F1 |
|---|---|---|---|---|---|---|
| Baseline CNN | 64.29% | 0.5535 | 0.5556 | 0.5833 | 0.2000 | 0.8750 |
| Improved CNN | 45.24% | 0.4189 | 0.4706 | 0.1667 | 0.5000 | 0.5385 |
| Fine-Tuned ResNet50 | **71.43%** | **0.7000** | **0.8000** | **0.6667** | **0.5333** | **0.8000** |

> The fine-tuned ResNet50 was also retrained with more aggressive class weighting and increased focus on leaf features, achieving **78.57%** real-world accuracy — an improvement in Late Blight (F1: 0.7619) and Healthy (F1: 0.6154) classification.

### Domain Gap Analysis

| Model | Standard | Real-World | Gap |
|---|---|---|---|
| Baseline CNN | 97.74% | 64.29% | 33.45% |
| Improved CNN | 96.71% | 45.24% | 51.47% |
| Fine-Tuned ResNet50 | 99.03% | 71.43% | **27.60%** |

The fine-tuned ResNet50 is selected as the best model for deployment — highest accuracy on both test sets and smallest domain gap. The improved CNN unexpectedly reduced real-world accuracy compared to the baseline, demonstrating that data augmentation alone cannot adequately bridge the domain gap between laboratory and field photography.

---

## Project Structure

```
plant_disease_detect/
│
├── notebooks/
│   ├── baseline_cnn.ipynb              # Model 1 — baseline CNN training and evaluation
│   ├── improved_cnn.ipynb              # Model 2 — improved CNN with augmentation
│   └── finetuned_resnet50.ipynb        # Model 3 — ResNet50 fine-tuning
│
├── diseaseDetection_UI.py              # Streamlit deployment UI
│
├── models/                             # Saved model checkpoints
│   ├── baseline_cnn_best.pth
│   ├── improved_cnn_best.pth
│   └── finetuned_resnet50_best.pth
│
├── combined_dataset/               # Combined PlantVillage + PlantDoc training data
│   └── train/
│       ├── Tomato___Early_blight/
│       ├── Tomato___Late_blight/
│       ├── Tomato___healthy/
│       └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│
├── realworld_test/                 # PlantDoc test set (for evaluation only)
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___healthy/
│   └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Clone the repository

```bash
git clone https://github.com/Apersonlived/plant_disease_detect.git
cd plant_disease_detect
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.22.0
tqdm>=4.65.0
opencv-python>=4.7.0
```

---

## Usage

### Dataset Setup

**1. Download PlantVillage**
```
https://github.com/spMohanty/PlantVillage-Dataset
```
Or via Kaggle: `https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset`

**2. Download PlantDoc**
```bash
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
```

> **Windows users:** The PlantDoc repository contains a filename with an invalid `?` character which causes checkout failure on Windows. Use the Kaggle version instead: `https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset`

**3. Build the combined dataset**

Run the dataset preparation cells in any notebook to handle folder merging, name standardisation, and train/test separation automatically. Images from PlantDoc are prefixed with `plantdoc_` to prevent filename collision.

### Training

Open and run the notebooks in order:

```bash
# Local Jupyter
jupyter notebook notebooks/baseline_cnn.ipynb
Remove the cells with these codes and replace them with your actual folder locations in your computer:
from google.colab import drive
drive.mount('/content/drive')

COMBINED_DIR       = "/content/drive/MyDrive/plant_disease/combined_dataset/train"
REALWORLD_TEST_DIR = "/content/drive/MyDrive/plant_disease/realworld_test"
```

> **Recommended:** Use Google Colab with T4 GPU. Training on CPU takes 15+ minutes per epoch. On T4 GPU each epoch takes 1–2 minutes.

**Save checkpoints to Drive immediately after training:**

```python
import shutil
shutil.copy("finetuned_resnet50_best.pth",
            "/content/drive/MyDrive/plant_disease/models/finetuned_resnet50_best.pth")
```

---

## Deployment

The UI is a web-based interface made using Streamlit. It contains a basic interface to select the model, upload an image, and get predictions with confidence levels for the top 3 classes.

### Setup

Place all three model checkpoints in the same folder as `diseaseDetection_UI.py`:

```
plant_disease_detect/
  diseaseDetection_UI.py
  baseline_cnn_best.pth
  improved_cnn_best.pth
  finetuned_resnet50_best.pth
```

### Run locally

```bash
streamlit run diseaseDetection_UI.py
```

Opens at `http://localhost:8501`

### Deployed version

The app is also deployed on Streamlit Community Cloud:  
`https://apersonlived-plant-disease-detect.streamlit.app`

### Features

- **Model selection** — choose between Baseline CNN, Improved CNN, or Fine-Tuned ResNet50 with accuracy stats for each
- **Image upload** — accepts JPG, JPEG, PNG formats
- **Prediction display** — predicted class with colour-coded result (green for Healthy, red/orange for disease)
- **Confidence score** — percentage and progress bar
- **Top-3 predictions** — ranked confidence breakdown for all top classes
- **Treatment plan** — disease description, severity level, and numbered treatment steps
- **Low confidence warning** — alerts when confidence drops below 60%

---

**Development environment:**
- Google Colab (T4 GPU) — model training
- Jupyter Notebook — local experimentation
- Visual Studio Code — deployment development
- Python 3.12 / Anaconda

---

## Challenges

**1. Domain gap between laboratory and real-world images**  
There is a significant gap between real-world and laboratory-controlled images causing the model to perform far worse in real-world tests. The gap proved significant enough to cause major challenges in bridging data availability differences between PlantVillage and PlantDoc.

**2. Class imbalance**  
Yellow Leaf Curl Virus accounts for nearly half the training data (5,580 of 10,319 images). Without intervention the model is biased toward the majority class. This was addressed through class-weighted loss functions in the improved and fine-tuned models.

**3. Computational constraints**  
Training on CPU proved impractical with a single epoch of the baseline model taking over 15 minutes on a local machine. All training was migrated to Google Colab with T4 GPU. Even with GPU acceleration, the improved CNN required reduction of input image size from 224×224 to 128×128 and batch size increase to 64 to achieve acceptable epoch times. GPU runtime limitations also resulted in delays and added further constraints.

**4. Dataset integration challenges**  
Combining PlantVillage and PlantDoc required manual normalisation of class folder names across different naming conventions. The PlantDoc GitHub repository also contained filenames with invalid characters, causing checkout failure on Windows file systems.

---

## Future Improvements

1. **Domain adaptation techniques** — CycleGAN and other image-to-image translation techniques could synthesise realistic field-style versions of PlantVillage images during training, giving the model exposure to real-world visual conditions without requiring additional manual annotations.

2. **Larger and more balanced real-world dataset** — the PlantDoc test set contains only 42 images across 4 classes, making per-class metrics highly sensitive to individual predictions. A larger field-captured test set would produce statistically meaningful real-world performance estimates.

3. **Attention mechanisms** — incorporating Squeeze-and-Excitation (SE) blocks or Convolutional Block Attention Module (CBAM) would allow the model to focus on leaf lesion regions rather than background features, directly addressing the domain gap problem.

4. **Ensemble methods** — combining predictions from all three models through weighted voting could yield higher accuracy than any individual model, especially for difficult real-world cases where models disagree.

5. **Expanded class coverage** — the current models cover only 4 tomato disease classes. Future work could extend to all tomato classes and additional crop species for a general-purpose multi-crop disease detection system.

6. **Mobile deployment** — converting the best-performing model to TensorFlow Lite or ONNX format for deployment as a mobile application, enabling farmers to use a smartphone camera for in-field diagnosis without internet connectivity.

7. **Severity grading** — future models could classify disease severity (early stage, moderate, severe) rather than just identifying the class, providing actionable information about the urgency of intervention required.

---

## References

Al-Kababji, A., Bensaali, F. & Dakua, S. P., 2022. Scheduling Techniques for Liver Segmentation: ReduceLRonPlateau Vs OneCycleLR. *arXiv*.

Amri, M. F. et al., 2023. Bacterial Classification Using Deep Structured Convolutional Neural Network for Low Resource Data. *Jurnal Elektronika dan Telekomunikasi*, 23(1), pp. 47–54.

Bhagat, S. et al., 2024. Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop. *Smart Agricultural Technology*, Volume 7.

GeeksforGeeks, 2026. What is Fully Connected Layer in Deep Learning. [Online] Available at: https://www.geeksforgeeks.org/deep-learning/what-is-fully-connected-layer-in-deep-learning/

Gurucharan, M., 2026. Basic CNN Architecture: How the 5 Layers Work Together. [Online] Available at: https://www.upgrad.com/blog/basic-cnn-architecture/

He, K., Zhang, X., Ren, S. & Sun, J., 2016. Deep Residual Learning for Image Recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.

Islam, M. T., 2020. Plant Disease Detection using CNN Model and Image Processing. *International Journal of Engineering Research & Technology*, 9(10), pp. 291–297.

KararHaiderDeepVision, 2025. ResNet-50 Explained Step by Step: The Easiest Guide to Deep Residual Networks. [Online] Available at: https://medium.com/@deepvisionkararhaider/resnet-50-explained-step-by-step-the-easiest-guide-to-deep-residual-networks-7616f4f45046

Li, L., Zhang, S. & Wang, B., 2021. Plant Disease Detection and Classification by Deep Learning — A Review. *IEEE Access*, 9(2169-3536), pp. 56683–56698.

msbrdmr, 2025. Impact of L1 and L2 regularisation with cross-entropy loss. [Online] Available at: https://stats.stackexchange.com/questions/576699/impact-of-l1-and-l2-regularisation-with-cross-entropy-loss

Pandian, J. A. et al., 2022. Plant Disease Detection Using Deep Convolutional Neural Network. *Applied Sciences*, 12(14), p. 6982.

Rosebrock, A., 2021. Convolutional Neural Networks (CNNs) and Layer Types. [Online] Available at: https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/

Shafay, M. et al., 2025. Recent advances in plant disease detection: challenges and opportunities. *Plant Methods*, 21(1).

Shrestha, G., Deepsikha, Das, M. & Dey, N., 2020. Plant Disease Detection Using CNN. *IEEE Applied Signal Processing Conference*.

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S. & Batra, N., 2020. PlantDoc: A Dataset for Visual Plant Disease Detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, pp. 249–253.

---

<div align="center">
  <p>Tomato Disease Detection · Deep Learning · PlantVillage + PlantDoc Dataset</p>
</div>
