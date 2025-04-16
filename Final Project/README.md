# Diabetes Prediction Using Machine Learning

## Overview

This project investigates the use of supervised machine learning models to predict diabetes based on patient data. Our goal is to develop a non-invasive diagnostic tool to help healthcare providers identify high-risk individuals efficiently. Multiple models were implemented and evaluated to determine the best performing approach.

Team: Dallas Grandy, John Hoffman  
Course: SAT5114  
Recorded Presentation ← (Insert Link Here)  
Dataset: [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)  
---

## Dataset Description

The dataset contains 100,000 patient records with the following features:

* gender: Male, Female, Other

* age: Numerical age of the patient

* hypertension: 1 if the patient has hypertension, 0 otherwise

* heart\_disease: 1 if the patient has heart disease, 0 otherwise

* smoking\_history: Categorical (current, not current, former, never, ever, no info)

* bmi: Body Mass Index

* HbA1c\_level: Hemoglobin A1c level

* blood\_glucose\_level: Blood glucose level at admission

* diabetes: Target variable (1 \= diabetic, 0 \= non-diabetic)

## Preprocessing Steps

Imbalance Handling: Noted class imbalance (91,500 non-diabetic vs 8,500 diabetic); stratified splitting used. Smoking history simplified by grouping similar categories as well as creating dummy variables for each One-hot encoding applied to gender and smoking\_history. Continuous numerical features were scaled using StandardScaler to normalize feature ranges since these variables were on different scales. This would prevent bias towards predictors with larger scales.

---

## Models Used

Each model was trained using a stratified 80-20 train-test split and evaluated with 5-fold cross-validation, optimizing for F1-score to address class imbalance.

* Logistic Regression

* Decision Tree

* Neural Network (MLPClassifier)

* Support Vector Machine (SVM)

* Random Forest

* K-Nearest Neighbors (KNN)

Hyperparameter tuning was performed using `GridSearchCV`

---

## 

## 

## Conclusion and Findings

After each model was tuned, they were evaluated on accuracy, class 1 and class 0 recall and precision, class 1 and class 0 F1 scores, their confusion matrix, and computation time. Our findings indicated that the Multi-Layer Perceptron was the best model for fast and accurate diabetes predictions. With an accuracy score of 0.97 and the highest recall for positive predictions being 0.81, 12% higher than any other model. Along with it’s relatively fast computation speed we can determine that the MLP model is the best performing and feasible model to bring into a clinical environment.