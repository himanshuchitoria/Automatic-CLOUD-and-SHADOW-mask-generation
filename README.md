# 🌥️ Automatic CLOUD & SHADOW Mask Generation

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 📌 Overview

This project implements a **U-Net based deep learning pipeline** for automatic pixel-wise segmentation of **clouds**, **cloud shadows**, **snow**, and **background** in **Resourcesat-2/2A LISS-4 satellite imagery**.

It was developed by our team as part of the **NRSC Offline Coding Challenge 2025**, offering end-to-end capabilities—training, inference, geospatial output generation, and performance evaluation.

📌 **Note:** This GitHub repository is created and maintained by our team independently. It is **not an official NRSC repository**.

---

## 📥 Dataset Source

The dataset used in this project was provided directly by **NRSC (National Remote Sensing Centre, ISRO)** via **secure SFTP access**. It includes:

- Multispectral LISS-4 satellite images  
- Ground truth masks for cloud, shadow, snow, and background  
- Sample folder structure for final submission

> 🔒 This dataset was exclusive to registered participants of the NRSC Coding Challenge and is **not publicly available**.

---

## 🧱 Key Features

- 🛰️ **Cloud & Shadow Segmentation**: Detects and masks clouds and their shadows.
- ❄️ **Snow & Background Classification**: Differentiates snow-covered areas and clear background.
- 🧠 **U-Net Architecture**: High-resolution semantic segmentation tailored to multispectral data.
- 🌍 **Georeferenced Outputs**:
  - Pixel masks as **GeoTIFFs**
  - **Shapefiles** for polygon exports
  - **Evaluation reports** with metrics (IoU, F1-score, etc.)
- 🚀 **Complete Workflow Support**: Training, validation, inference, and evaluation.

---

## 🗂️ Repository Structure

![image](https://github.com/user-attachments/assets/2fecda64-ecf3-4576-b87e-7c934aa6c19f)

---

## 🧠 How It Works

This project utilizes a **U-Net** architecture to perform **semantic segmentation** on satellite imagery for detecting **clouds**, **shadows**, **snow**, and **background**. Below is a step-by-step breakdown of how the system works:

---

### 🏗️ U-Net Model

- Implements a **deep encoder-decoder** architecture.
- Incorporates **skip connections** between encoder and decoder layers to retain spatial information.
- Suitable for **pixel-wise segmentation** tasks with high precision.

---

### 🎓 Training Phase

- **Loss Function**:  
  Uses **Categorical Cross-Entropy** or optionally **Dice Loss** to optimize segmentation accuracy.

- **Data Augmentation**:  
  Random rotations, flips, and brightness adjustments are applied to improve generalization.

- **Monitoring**:  
  Tracks **validation loss** after each epoch and saves the **best model weights**.

---

### 🔍 Inference & Geo-Output

- The trained model predicts **class probabilities per pixel** for each test image.
- The predicted masks are saved as **GeoTIFFs**, preserving geospatial metadata.
- **Connected Component Labeling** is applied to:
  - Extract and classify **polygon shapes**
  - Export them as **shapefiles** for each class (Cloud, Shadow, Snow)

---

### 📊 Evaluation Phase

- The predicted masks are compared with ground truth masks.
- Computes standard evaluation metrics:
  - **IoU (Intersection over Union)** for each class.
  - **F1-score** for assessing segmentation quality.
- All evaluation results are saved in JSON or tabular format under the `/reports` folder.

---

This pipeline provides a complete framework for robust, georeferenced segmentation of satellite images and is fully compatible with **Resourcesat-2/2A LISS-4** data.

---
