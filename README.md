# AI Egg Fertility Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org)
[![Flutter](https://img.shields.io/badge/Flutter-Mobile_App-blue.svg)](https://flutter.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Automated deep learning system for classifying fertile vs. infertile chicken eggs using CNNs. Achieves 96% accuracy with real-time mobile deployment.

---

## Overview

A production-ready AI solution for the poultry industry, replacing manual candling with fast, accurate computer vision. Features transfer learning and mobile optimization for offline, real-time inference.

**Highlights:**  
- 96% accuracy  
- <0.5s inference  
- 30 FPS real-time detection  
- Fully offline operation  

## Tech Stack

```bash
Backend:    TensorFlow 2.10, OpenCV 4.7, Keras
Mobile:     Flutter, TensorFlow Lite, NNAPI
Training:   Google Colab Pro (Tesla T4), Roboflow
Dataset:    1,412 images (augmented)
```

## Architecture

<img src="https://github.com/T0MM11Y/AI-Egg-Fertility-Classification/blob/main/initialAndModelArch.png" alt="CNN Architecture" width="430">

**Sequential CNN:**  
- Conv2D(32) → MaxPool  
- Conv2D(64) → MaxPool  
- Conv2D(128) → MaxPool  
- Flatten → Dense(128) → Dense(1, sigmoid)  
- **Params:** 4,828,481

**Training:**  
- Optimizer: Adam (lr=0.001)  
- Loss: BinaryCrossentropy  
- Epochs: 20, Batch: 32  
- Augmentation: Rotation, Shear, Zoom  
- EarlyStopping, ModelCheckpoint

## Results

<img src="https://github.com/T0MM11Y/AI-Egg-Fertility-Classification/blob/main/resultClassification.png" alt="Results" width="480">

```
Accuracy: 96.0%   Precision: 96.3%
Recall:   96.3%   F1-Score: 96.3%

Confusion Matrix:
           Fertile  Infertile
Fertile      26         1
Infertile     1        22
```

## Quick Start

```bash
git clone https://github.com/T0MM11Y/AI-Egg-Fertility-Classification.git
cd AI-Egg-Fertility-Classification
pip install -r requirements.txt
# For mobile:
flutter pub get && flutter run
# For desktop:
python main.py
```

## Project Structure

```
├── model/                # Trained models (.h5, .tflite)
├── mobile_app/           # Flutter app
├── notebooks/            # Training pipeline
├── src/                  # Core logic
└── dataset/              # Images
```

## Mobile App

- Real-time camera & gallery detection
- Confidence scoring
- Fully offline
- **Android 7.0+, 2GB RAM, NNAPI**

**Demo Video:** [Watch Here](https://drive.google.com/file/d/1Fl-cpigO6duf8Nz68-Gum1jm7R6nuDgm/view?usp=drivesdk)

---

**MIT License** | [Institut Teknologi Del](https://www.del.ac.id) | Laguboti, North Sumatera
