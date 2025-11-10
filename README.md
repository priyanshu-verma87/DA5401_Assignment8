# **DA5401 Assignment 8 — Ensemble Learning for Complex Regression Modeling on Bike Share Data**  

**Name:** Priyanshu Verma  
**Roll No:** CH22B087  
**Date:** 10-11-2025  

---

## **Project Overview**  

This assignment explores the application of **ensemble learning techniques** to a complex **bike rental demand forecasting** problem using the **UCI Bike Sharing Demand Dataset**. The objective is to predict the total hourly count of rented bikes (`cnt`) based on various temporal, weather, and seasonal features that exhibit non-linear relationships and high variability.  

Three ensemble paradigms - **Bagging**, **Boosting**, and **Stacking** are implemented and compared against baseline single models (**Decision Tree Regressor** and **Linear Regression**) to demonstrate how ensemble learning improves prediction accuracy by addressing the **bias–variance trade-off**.  

The main deliverable is a Jupyter Notebook (`DA5401_Assignment_8.ipynb`) containing detailed preprocessing, model training, evaluation, and interpretive analysis.

---

## **Files in this Repository**  

- `DA5401_Assignment_8.ipynb` — main notebook (complete implementation).  
- `README.md` — this documentation file.  

---

## **Notebook Structure**  

The notebook is organized into the following sections:  

1. **Assignment Description**: Overview of the problem, dataset, and objectives.  
2. **Part A: Data Preprocessing and Baseline Models**  
   - Load and preprocess the Bike Sharing dataset.  
   - Apply One-Hot Encoding to categorical features.  
   - Train and evaluate **Decision Tree Regressor (max_depth = 6)** and **Linear Regression** using RMSE.  
3. **Part B: Ensemble Techniques for Bias and Variance Reduction**  
   - Implement **Bagging Regressor** (variance reduction) using the Decision Tree base estimator.  
   - Implement **Gradient Boosting Regressor** (bias reduction) and compare results.  
4. **Part C: Stacking for Optimal Performance**  
   - Explain stacking principles and define base learners (**KNN**, **Bagging**, **Gradient Boosting**) and meta-learner (**Ridge Regression**).  
   - Implement **Stacking Regressor** and compute RMSE on the test set.  
5. **Part D: Final Analysis**  
   - Create a **comparative RMSE table** for all models.  
   - Discuss results in the context of the bias–variance trade-off and model diversity.  
6. **Conclusion**: Identify the best-performing model and summarize key learnings about ensemble effectiveness.  

---


**Final Insight:**  
The **Stacking Regressor** outperformed all other models, achieving the lowest RMSE and the best generalization performance. This supports the hypothesis that combining diverse learners through meta-learning can yield superior predictive accuracy compared to individual ensemble methods like Bagging or Boosting alone.

---

## ⚙️ **Requirements**

Recommended environment:  
- Python 3.8+ (3.9/3.10 recommended)  
- JupyterLab or Jupyter Notebook  

### **Required Python Packages**
Install the following packages before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
