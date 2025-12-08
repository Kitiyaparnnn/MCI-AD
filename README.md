# 🧠 Transcriptomic Biomarkers for Predicting MCI-to-Alzheimer’s Progression

Computational Medicine — Final Project
Authors: Kitiyaporn Takham, Mahitha Chaturvedula

## 📌 Project Overview

This repository contains all the code used in our computational medicine project:
Identifying transcriptomic biomarkers that distinguish stable MCI from progressive MCI and predicting progression to Alzheimer’s dementia using machine learning and deep learning.

We evaluate three supervised models:

- Linear Support Vector Machine (SVM)

- 1D Convolutional Neural Network (1D-CNN)

- 2D Convolutional Neural Network (2D-CNN)

We also extract biomarkers from each model using:

- SVM coefficients

- Integrated Gradients (for CNNs)

External validation is performed using an independent GEO cohort.

## 🧪 Data Sources
**Training Cohort**

- GSE282742
- Blood transcriptomics (TPM)
- Progressive MCI (n = 28) vs Stable MCI (n = 39)

**External Cohort**

- GSE249477
- Blood transcriptomics (TPM)
- MCI due to AD (n = 20) vs AD (n = 21)

**Preprocessing**
- Intersection of shared genes: 21,462 genes
- Z-score normalization for ML models
- Min-max scaling for CNN models

## 🧬 Biomarker Identification Summary

Across all three models, two genes consistently ranked highly:
- FAM118A
- IGKV2D-29

These may serve as robust blood-based biomarkers for identifying MCI individuals at higher risk for progression to Alzheimer’s dementia.

## 📊 Model Performance Comparison

Below is the overall comparison of model performance evaluated on the external validation cohort (GSE249477).
| **Model**      | **Accuracy** | **Sensitivity**   | **Specificity** | 
| -------------- | ------------ | ----------------- | --------------- | 
| **Linear SVM** | 41.46%       | 80.95%            | 0%              | 
| **1D-CNN**     | 53.66%       | 23.81%            | 85%             |
| **2D-CNN**     | 48.78%       | 0% | 100%        | 
