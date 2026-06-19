# 🩺 AI-Powered Pneumonia Detection System

## 📌 Overview

AI-Powered Pneumonia Detection System is a Deep Learning project that classifies Chest X-ray images as **Normal** or **Pneumonia** using the DenseNet121 architecture.

The project includes an interactive Streamlit web application that allows users to upload Chest X-ray images and receive real-time predictions with confidence scores.

---

## 🎯 Key Features

* Chest X-ray Image Classification
* Pneumonia Detection using Deep Learning
* DenseNet121 Transfer Learning Model
* Real-Time Prediction
* Confidence Score Display
* Interactive Streamlit Web Application

---

## 🧠 Model Architecture

* DenseNet121 (Pre-trained)
* Global Average Pooling Layer
* Dense Layer (128 Units)
* Dropout Layer
* Binary Classification Output Layer

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 95.80% |
| Precision | 94.55% |
| Recall    | 97.20% |
| AUC Score | 99.07% |

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* NumPy
* Scikit-Learn
* Pillow
* Matplotlib

---

## 📂 Project Structure

```text
AI-Powered-Pneumonia-Detection-System/
│
├── app.py
├── train.py
├── metrics.py
├── requirements.txt
├── README.md
│
└── project_images/
    ├── home_page.png
    ├── pneumonia_prediction.png
    └── normal_prediction.png
```

---

## 📸 Application Screenshots

### Home Page

Add screenshot here.

### Pneumonia Detection

Add screenshot here.

### Normal Detection

Add screenshot here.

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone <your-repository-link>
```

### Navigate to the Project Folder

```bash
cd AI-Powered-Pneumonia-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📁 Dataset

This project was trained using a subset of the Chest X-Ray Pneumonia dataset.

Dataset Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Due to GitHub repository size limitations, the dataset is not included in this repository.

---

## ❗ Important Note

The trained model file (`densenet_pneumonia.h5`) is not included in this repository because of GitHub file size limitations.

Users can train the model using `train.py` and generate their own model file.

---

## 🔮 Future Improvements

* Grad-CAM Explainability
* PDF Report Generation
* Multi-Disease Detection
* Cloud Deployment
* Prediction History Tracking

---

## 👩‍💻 Author

**Harini D**

AI & Data Science Student

---

⭐ If you found this project useful, consider giving it a star.
