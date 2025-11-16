# 🧠 Brain Tumor Classification — Streamlit Web App

This project is a **deep learning–based Brain Tumor Classification System** built using **TensorFlow** and deployed with **Streamlit**.  
Users can upload MRI images, and the model predicts the tumor category using a trained Convolutional Neural Network (CNN).

---

## 🚀 Features

- Upload MRI brain images (JPG/PNG)
- Automatic preprocessing (resize, normalization, RGB conversion)
- Deep learning–based predictions using a custom CNN model
- Dockerized for easy deployment
- Clean and interactive Streamlit UI

---

## 🏗️ Project Structure

```bash
.
├── app.py
├── cnn_11_layer.h5
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Model Details

- Custom CNN built using **TensorFlow/Keras**
- Input shape: **224 × 224 × 3**
- Multiple Conv2D + MaxPooling layers
- Fully connected layers with Dropout for regularization
- Trained on MRI brain tumor dataset
- Saved as `cnn_11_layer.h5`

---

## 🌐 Running the App (Locally)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 🔧 Tech Stack

- TensorFlow / Keras
- Streamlit
- Docker
- NumPy
- Pillow (PIL)
- Python 3.13
