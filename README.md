# Face Recognition Attendance System

## Overview
A complete 3-stage face recognition pipeline that automatically marks 
attendance by identifying individuals in real-time using a webcam.
Built from scratch — from data collection to live deployment.

## Pipeline

### Stage 1 — create_dataset.ipynb
- Captures face photos from webcam for each person
- Uses Haar Cascade (haarcascade_frontalface_default.xml) for face detection
- Crops and resizes faces to 50x50 pixels
- Saves processed data as data.npy and labels as target.npy

### Stage 2 — train_data.ipynb
- Loads face data from data.npy and target.npy
- Applies PCA (86 components, whitened) for dimensionality reduction
- Trains a Linear SVM classifier using a scikit-learn Pipeline
- Evaluates model performance:
  - Accuracy: 65.2% on test set
  - Generates classification report and confusion matrix
- Saves trained model as face_recognized_svm.sav using joblib

### Stage 3 — test_data.ipynb
- Loads trained PCA + SVM model
- Opens live webcam feed
- Detects faces in real-time using Haar Cascade
- Predicts identity using the trained model
- Marks attendance with timestamp in Get_Attendence.csv
- Displays recognized name on screen in real-time

## People Recognized
The system was trained to recognize 4 individuals:
devindi, kavindya, mindinu, rusiru

## Model Details
- Face Detection: Haar Cascade Classifier (OpenCV)
- Dimensionality Reduction: PCA (n_components=86, whiten=True)
- Classifier: Support Vector Machine — Linear Kernel (scikit-learn)
- Pipeline: PCA → SVM (scikit-learn Pipeline)
- Train/Test Split: 80/20
- Test Accuracy: 65.2%

## Why PCA + SVM?
Face images (50x50 = 2500 features) have high dimensionality.
PCA reduces this to 86 meaningful components before SVM classification,
which improves speed and reduces overfitting on small datasets.

## Tech Stack
- Python 3.9
- OpenCV (face detection, webcam, image processing)
- scikit-learn (PCA, SVM, Pipeline, metrics)
- NumPy (data storage and manipulation)
- joblib (model saving/loading)
- Jupyter Notebook

## How to Run
1. Clone this repository
2. Install requirements:
   pip install opencv-python scikit-learn numpy joblib pandas
3. Create a dataset/ folder with subfolders for each person
4. Run create_dataset.ipynb — captures and saves face data
5. Run train_data.ipynb — trains PCA+SVM model
6. Run test_data.ipynb — starts live recognition and attendance marking

## Note on Dataset
The dataset/ folder is not included to protect the privacy
of individuals photographed during development.

## Author
Kavindya Hewage | Abu Dhabi, UAE
LinkedIn: linkedin.com/in/kavindya-sureshi-8b39681b8/
Portfolio: https://kavindya-hewage.github.io
