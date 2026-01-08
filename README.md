# pneumonia-xray-classification-cnn-inceptionv3

## 1. Project Overview
This project implements a medical image classification system based on Convolutional Neural Networks (CNNs) for pneumonia detection using chest X-ray images.

Two approaches are studied:
- A CNN trained from scratch
- A transfer learning model based on InceptionV3

## 2. Dataset
The dataset is the Chest X-ray Pneumonia dataset from Kaggle, organized as:

chest_xray/
├── train/
├── val/
└── test/
Each folder contains two classes: NORMAL and PNEUMONIA.

## 3. Project Structure
- data_explore.py: exploratory data analysis
- train_cnn.py: CNN training and evaluation
- train_inception.py: transfer learning with InceptionV3
- plot_utils.py: visualization utilities

## 4. Environment Setup
Install dependencies using:
pip install -r requirements.txt

## 5. Running the Code
1. Data analysis:
   python data_explore.py

2. Train CNN from scratch:
   python train_cnn.py

3. Train InceptionV3 model:
   python train_inception.py

## 6. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

## 7. Application
This system demonstrates how deep learning can assist in automated pneumonia diagnosis using medical imaging.
