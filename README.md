# AI Egg Fertility Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org)
[![Flutter](https://img.shields.io/badge/Flutter-Mobile_App-blue.svg)](https://flutter.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Deep Learning-powered automated classification system for distinguishing fertile and infertile chicken eggs using CNN architecture. Achieves 96% accuracy with real-time mobile deployment.

## Overview

This project implements an end-to-end computer vision solution for poultry industry automation, replacing traditional candling methods with AI-driven classification. The system leverages transfer learning and mobile optimization for production-ready deployment.

**Key Achievements**: 96% accuracy | <0.5s inference | Offline capability | 30 FPS real-time detection

## Technical Stack

```bash
Backend:    TensorFlow 2.10, OpenCV 4.7, Keras
Mobile:     Flutter, TensorFlow Lite, NNAPI
Training:   Google Colab Pro (Tesla T4), Roboflow
Dataset:    1,412 augmented images (699 fertile, 663 infertile)
```

## Architecture

<img src="https://github.com/T0MM11Y/AI-Egg-Fertility-Classification/blob/main/initialAndModelArch.png" alt="CNN Architecture" width="450">

### Model Specifications
```python
Sequential CNN Architecture:
├── Conv2D(32, 3x3) → MaxPool2D → (74×74×32)
├── Conv2D(64, 3x3) → MaxPool2D → (36×36×64)  
├── Conv2D(128, 3x3) → MaxPool2D → (17×17×128)
├── Flatten → Dense(128) → Dense(1, sigmoid)
└── Total Parameters: 4,828,481
```

### Training Configuration
```yaml
Optimizer: Adam (lr=0.001)
Loss: BinaryCrossentropy
Epochs: 20 | Batch: 32
Callbacks: EarlyStopping, ModelCheckpoint
Augmentation: Rotation(±15°), Shear(±10°), Zoom(20%)
```

## Performance Metrics

<img src="https://github.com/T0MM11Y/AI-Egg-Fertility-Classification/blob/main/resultClassification.png" alt="Results" width="500">

```
Accuracy: 96.0%     Precision: 96.3%
Recall: 96.3%       F1-Score: 96.3%

Confusion Matrix:
           Fertile  Infertile
Fertile      26       1
Infertile     1      22
```

## Quick Start

```bash
# Clone & Setup
git clone https://github.com/T0MM11Y/AI-Egg-Fertility-Classification.git
cd AI-Egg-Fertility-Classification
pip install -r requirements.txt

# Mobile Deployment
flutter pub get
flutter run

# Desktop Testing
python main.py
```

## Project Structure

```
├── model/
│   ├── egg_classifier.h5      # TensorFlow model
│   └── egg_classifier.tflite  # Mobile-optimized model
├── mobile_app/                # Flutter application
├── notebooks/training.ipynb   # Model development
├── src/                       # Core implementation
└── dataset/                   # Training data
```

## Mobile Application

**Features**: Real-time camera detection | Gallery analysis | Confidence scoring | Offline operation

**Requirements**: Android 7.0+ | 2GB RAM | NNAPI support

**Preview Implementation**: [Demo APK](https://drive.google.com/file/d/1Fl-cpigO6duf8Nz68-Gum1jm7R6nuDgm/view?usp=drivesdk)

- Dataset: Eggs3 v1 (Roboflow Universe)

---

**MIT License** | Contact: [Institut Teknologi Del](https://www.del.ac.id) | Laguboti, North Sumatera
