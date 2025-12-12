# 🧠 Transcriptomic Biomarkers for Predicting MCI-to-Alzheimer’s Progression

Computational Medicine — Final Project
Authors: Kitiyaporn Takham, Mahitha Chaturvedula

## 📌 Project Overview

This repository contains all the code used in our computational medicine project:
Identifying transcriptomic biomarkers that distinguish stable MCI from progressive MCI and predicting progression to Alzheimer’s dementia using machine learning and deep learning.

We evaluate three supervised models:

- Linear Support Vector Machine (SVM)
- Logistic Regression (LR)
- 1D Convolutional Neural Network (1D-CNN)

We also extract a set of gene biomarkers from each model using:

- Leave-one-out (LOOCV) (for SVM and LR)

- Integrated Gradients (for CNNs)

## 🧪 Data Sources
**Training Cohort**

- GSE282742
- Blood transcriptomics (TPM)
- Progressive MCI (n = 28) vs Stable MCI (n = 39) vs AD (n = 49)

**External Cohort**

- GSE249477
- Blood transcriptomics (TPM)
- MCI due to AD (n = 20) vs AD (n = 21)

**Preprocessing**
- Intersection of shared genes: 21,462 genes
- Z-score normalization for ML models
- Mutual Information Filter to keep the top initial 2,000 genes 

## 🧬 Biomarker Identification Summary

Across all three models, several MCI/AD-related genes were used as biomarkers, including
CASP7, COL4A1, GLB1, PPARG, PON1, and CXCL8.

## 📊 Model Performance Comparison

- MCI vs. AD classification

| **Model** | **Train Accuracy** | **Evaluation Accuracy** |
| -------------- | ------------ |  ------------ | 
| **Linear SVM (80 genes)** | 99.14%       | 53.66%|
| **Logistic Regression (120 genes)** | 100% | 56.10%|
| **1D-CNN (100genes)**     | 59.30%      | 39.02% |


- S-MCI vs. P-MCI vs. AD classification

| **Model** | **Train Accuracy** | **Evaluation Accuracy** |
| -------------- | ------------ |  ------------ | 
| **Linear SVM (160 genes)** | 82.80%       | 56.10%|
| **Logistic Regression (190 genes)** | 90.50% | 53.66%|
| **1D-CNN (100 genes)**     | 42.14%      | 36.59% |
