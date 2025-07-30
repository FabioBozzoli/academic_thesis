# Support Vector Machines vs. Ensemble Methods: Comparative Study and Applications in Medical Data

**Author:** Fabio Bozzoli  

**Degree:** Bachelor of Science in Computer Engineering  

**University:** Unicusano  

**Original Language:** Italian  

*Note: The full thesis document is provided in Italian. This README offers an English summary and context for the included code.*

---

## Overview

This project presents my undergraduate thesis, which focuses on the comparative analysis and practical application of Support Vector Machines (SVM) and AdaBoost ensemble methods for the diagnosis of cardiovascular diseases using real-world medical data.

The repository includes:
- The full thesis document (`engineering_thesis_fabiobozzoli.pdf`, in Italian)
- Two custom Python implementations:
    - `class_svm_py.py` — SVM classifier with probability calibration via logistic regression
    - `class_adaboost.py` — AdaBoost classifier with decision tree base learners

---

## Objectives

- Develop and compare SVM (both linear, coded from scratch, and kernel-based) and AdaBoost algorithms for medical data classification.
- Assess the strengths and weaknesses of each method in terms of predictive power, reliability, and interpretability, especially for heart disease diagnosis.
- Provide hands-on, transparent and well-documented Python implementations for educational and research purposes.

---

## Methodology

- **Data:** Heart disease dataset (917 samples, 11 features, binary target).
- **Preprocessing:** Handling missing values, categorical encoding (Label/OneHot/Ordinal), feature scaling, stratified train-test split.
- **Custom Models:**  
    - `class_svm_py.py` implements a linear SVM with SGD + Platt scaling for calibrated probabilities.
    - `class_adaboost.py` implements classic AdaBoost with decision stumps.
- **Evaluation:** Accuracy, Precision, Recall, F1-score, AUC-ROC, and confusion matrices, following best practices for clinical data analysis.

---

## Key Findings

- AdaBoost outperformed SVM in most evaluation metrics, proving more robust for this task.
- From-scratch implementations closely matched the performance of `scikit-learn`'s optimized models, demonstrating a clear understanding of algorithm mechanics.
- The Python code is provided for full transparency and future educational use.

---

## Contents

- `engineering_thesis_fabiobozzoli.pdf`: Full thesis document (Italian)
- `class_svm_py.py`: Full Python implementation of a linear SVM with probability calibration.
- `class_adaboost.py`: Full Python implementation of AdaBoost classifier.
- `[Optional] code/` : Jupyter notebooks and scripts for data processing, analysis, and result visualization.

---

## Usage

You can use the Python classes provided to train/test SVM and AdaBoost models on your own datasets, or to replicate/extend the experiments performed in the thesis. For example:

```python
from class_svm_py import SVM
from class_adaboost import AdaBoost

# SVM and AdaBoost usage as described in docstrings and thesis
```
See each `.py` file for details and example usage.

---

**Disclaimer:**  
The full thesis is in Italian. An English summary is provided here and the code/documentation is intended to be accessible to an international audience.

---
