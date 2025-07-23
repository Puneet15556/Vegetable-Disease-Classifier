# Vegetable-Disease-Classifier
# 🥦 Plant Disease Classification using Deep Learning

A deep learning-powered image classification system to detect diseases in plant-based vegetables — **Potato, Tomato, and Pepper** — using both a custom CNN architecture and a ResNet50-based transfer learning model. The solution is deployed as a Streamlit web app to enable real-time predictions with confidence scores.

---

## 🚀 Project Overview

This project aims to assist farmers and agriculturists by providing an automated tool for identifying plant diseases using image inputs. Leveraging state-of-the-art deep learning techniques, the system classifies disease states in commonly grown vegetables.

---

## 🌿 Features

- 📸 **Image-based Disease Detection** for:
  - Potato
    


- 🧠 **Model Approaches**:
  1. ✅ **Custom CNN Architecture** – Built from scratch using convolutional layers, pooling, dropout, and dense layers.
  2. 🔁 **Transfer Learning (ResNet50)** – Leveraged pre-trained weights on ImageNet for robust feature extraction and fine-tuning.

- 🔄 **Data Augmentation** – Implemented rotation, flipping, zooming, and more to enhance generalization.

- 🌐 **Web App Deployment** – Built with **Streamlit**, allowing users to upload an image and get:
  - Predicted disease class
  - Model confidence score
  - Visualization of uploaded image

---

## 🧠 Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| Python | Programming Language |
| TensorFlow / Keras | Deep Learning Framework |
| OpenCV / PIL | Image Preprocessing |
| Matplotlib / Seaborn | Visualization |
| Streamlit | Web App Deployment |
| ResNet50 | Pre-trained Transfer Learning Model |

---

## 🛠️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/plant-disease-classifier.git
   cd plant-disease-classifier

