# Thesis: Machine Learning Models Based on Image Statistics for Visual Quality Quantification

**Author:** Fabio Bozzoli

**Degree:** Bachelor of Science in Mathematics

**University:** University of Modena and Reggio Emilia

**Original Language:** Italian *(Note: The full thesis document is in Italian. This README provides a comprehensive English summary.)*

## Abstract

This thesis explores the application of machine learning techniques for No-Reference Image Quality Assessment (NR-IQA), a field dedicated to quantifying the perceptual quality of an image without a reference "perfect" image. The core of this work focuses on leveraging the **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)** algorithm to extract a set of 36 statistical features from images. These features, which describe the image's naturalness, are then used as input for supervised learning models to predict a quality score.

Two primary machine learning models are implemented, analyzed, and compared: **Support Vector Regression (SVR)** and **Random Forest**. The study delves into the theoretical foundations of both models, including the principles of Support Vector Machines, kernel methods for non-linear cases, decision trees, and ensemble methods like bagging.

The models were trained and evaluated on the **Helsinki Deblur Challenge dataset**, where image quality is defined by the character recognition rate of an OCR system. The final results demonstrate that both SVR and Random Forest models can effectively predict image quality scores, with a detailed analysis of their performance based on hyperparameter tuning (e.g., kernel type and C for SVR, number of trees for Random Forest). The comparative study concludes that both models achieve similar predictive accuracy, with SVR offering a significant advantage in terms of training time.

## Key Contributions

*   **In-depth Theoretical Review:** Provides a comprehensive mathematical review of Support Vector Regression and Random Forest algorithms, including decision tree splitting criteria (Gini, Information Gain), overfitting mitigation (pruning), and ensemble logic (bagging).
*   **BRISQUE Feature Extraction:** Implements a pipeline to process images and extract the 36-feature BRISQUE vector, which captures statistical deviations from natural images. This method effectively reduces a high-dimensional image problem into a fixed-size feature vector, making it suitable for standard ML models.
*   **Comparative Model Analysis:** Conducts a rigorous experimental comparison between SVR (with linear, polynomial, and RBF kernels) and Random Forest models. The analysis evaluates performance using key regression metrics like **R²**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**.
*   **Hyperparameter Impact Study:** Investigates the sensitivity of the models to different hyperparameters, demonstrating the trade-offs between model complexity, overfitting, and generalization performance. The study shows that linear SVR and Random Forest with a sufficient number of trees yield the best and most stable results for this task.

## Methodology

1.  **Dataset**: The Helsinki Deblur Challenge dataset was used, which includes images with varying levels of blur and corresponding OCR-based quality scores.
2.  **Feature Engineering**: The BRISQUE algorithm was applied to each image to compute a 36-dimensional feature vector based on Mean Subtracted Contrast Normalization (MSCN) coefficients and their statistical properties.
3.  **Model Training**:
    *   **Support Vector Regression (SVR)**: Trained with different kernels (linear, polynomial, RBF) and regularization parameters (`C`) to find the optimal configuration.
    *   **Random Forest**: Trained with a varying number of decision trees to observe the impact on performance.
4.  **Evaluation**: Models were evaluated on a dedicated test set. The coefficient of determination (R²) and Mean Absolute Error (MAE) were the primary metrics used to compare the predictive accuracy.

## Results

The experiments demonstrated that:
*   A linear SVR model with `C≈1` achieved an R² score of **59.19%** on the test set.
*   A Random Forest model with 1000 trees achieved a comparable R² score of **59.23%**.
*   While both models showed similar predictive power, the **SVR model had a significantly lower training time**, making it a more efficient choice for this specific problem.

## Technologies Used

*   **Primary Language:** MATLAB
*   **Core Concepts:** Machine Learning, Image Quality Assessment (IQA), Support Vector Regression (SVR), Random Forest, Feature Extraction (BRISQUE)
*   **Libraries/Toolkits:** MATLAB Statistics and Machine Learning Toolbox

## How to Access

*   **Full Thesis Document (Italian):** [Link to the PDF file in your repository, e.g., `mathematics_thesis_fabiobozzoli.pdf`]
