# %%
# Import Required Libraries for data visualization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import os for directory path
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Import time to evaluate model compilation
import time
# %%
# Read in the Data
diabetes = pd.read_csv('diabetes_dataset.csv')

### DEBUG ###
# print(diabetes.head())
# print(len(diabetes))

# %%
# Print Basic Predictor Visualization
# Gender Visual
gender_plot = diabetes['gender'].value_counts().plot(kind = 'bar', title = 'Gender', xlabel = '', ylabel = 'Frequency',
                                                     color = ['pink', 'blue', 'green'], edgecolor = 'black')
# Print Numbers above Bar Chart
for container in gender_plot.containers:
    gender_plot.bar_label(container)
plt.show()

# Age Visual
age_plot = diabetes['age'].plot(kind = 'hist', edgecolor = 'black', bins = [0, 10, 20, 30, 40, 50, 60, 70, 80],
                                title = 'Age Range', xlabel = 'Age', ylabel = 'Frequency')
for container in age_plot.containers:
    age_plot.bar_label(container)
plt.show()

# Hypentension Visual
hyper_plot = diabetes['hypertension'].value_counts().plot(kind = 'bar', color = ['green', 'red'], title = 'Hypertension',
                                                          xlabel = 'Diagnosis', ylabel = 'Frequency', edgecolor = 'black')
hyper_plot.set_xticklabels(['No', 'Yes'])
for container in hyper_plot.containers:
    hyper_plot.bar_label(container)
plt.show()

# Heart Disease Visual
heart_plot = diabetes['heart_disease'].value_counts().plot(kind = 'bar', color = ['green', 'red'], title = 'Heart Disease',
                                                           xlabel = 'Diagnosis', ylabel = 'Frequency', edgecolor = 'black')
heart_plot.set_xticklabels(['No', 'Yes'])
for container in heart_plot.containers:
    heart_plot.bar_label(container)
plt.show()

# Smoking History Visual
smoking_plot = diabetes['smoking_history'].value_counts().plot(kind = 'bar', title = 'Smoking History',
                                                           xlabel = 'Response', ylabel = 'Frequency', edgecolor = 'black')
for container in smoking_plot.containers:
    smoking_plot.bar_label(container)
plt.show()

# BMI Visual
bmi_plot = diabetes['bmi'].plot(kind = 'hist', bins = range(10, 100, 10), title = 'Body Mass Index',
                                xlabel = 'Body Mass Value', ylabel = 'Frequency', edgecolor = 'black')
for container in bmi_plot.containers:
    bmi_plot.bar_label(container, fontsize = 8)
plt.show()

# Hemoglobin A1c Visual
hemo_plot = diabetes['HbA1c_level'].plot(kind = 'hist', bins = np.arange(3.5, 9, 0.5), title = 'Hemoglobin A1c Levels',
                                         xlabel = 'Hemoglobin Levels', ylabel = 'Frequency', edgecolor = 'black')
plt.xticks(np.arange(3.5, 9, 0.5))
for container in hemo_plot.containers:
    hemo_plot.bar_label(container, fontsize = 9)
plt.show()

# Blood Glucose Visual
bg_plot = diabetes['blood_glucose_level'].plot(kind = 'hist', bins = range(80, 300, 20), title = 'Blood Glucose',
                                               xlabel = 'BG Levels', ylabel = 'Frequency', edgecolor = 'black')
plt.xticks(range(80, 300, 20))
for container in bg_plot.containers:
    bg_plot.bar_label(container, fontsize = 9)
plt.show()

# Diabetes Visual
diabetes_plot = diabetes['diabetes'].value_counts().plot(kind = 'bar', title = 'Diabetes', xlabel = 'Diagnosis',
                                                         ylabel = 'Frequency', edgecolor = 'black', color = ['green', 'red'])
diabetes_plot.set_xticklabels(['No', 'Yes'])
for container in diabetes_plot.containers:
    diabetes_plot.bar_label(container)
plt.show()

#%%
# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# %%
# Preprocess smoking, combine some of them and one hot encode
diabetes['smoking_history'] = diabetes['smoking_history'].replace({
    'not current': 'former',
    'ever': 'former'
})

# Then encode
diabetes = pd.get_dummies(diabetes, columns=['smoking_history'], prefix='smoke', drop_first=True)

# One hot encode gender
diabetes = pd.get_dummies(diabetes, columns=['gender'], drop_first=True)

#%%
# Scale the data
scaler = StandardScaler()
continuous = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
diabetes[continuous] = scaler.fit_transform(diabetes[continuous])

#%%
# Data Splitting
X = diabetes.drop(columns='diabetes')
y = diabetes['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#%%
# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

start_time = time.perf_counter()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#%%
# Decision Tree Model
# Define hyperparameter grid
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize model
dt_model = DecisionTreeClassifier(random_state=42)

start_time = time.perf_counter()

# Perform grid search with F1-score as scoring metric
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best model from grid search
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Print results
print("Best Parameters:")
print(grid_search.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% 
# Multi-Layer Perceptron Model
# Perform Grid Search
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
    'solver': ['adam'],
    'max_iter': [300]
}

start_time = time.perf_counter()

mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
# Support Vector Machine Model
# Perform Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 4]  # only used by 'poly' kernel
}

start_time = time.perf_counter()

svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
# Random Forest Model
# Perform Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

start_time = time.perf_counter()

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
# K-NN Classification model
# Perform Grid Search
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
}

start_time = time.perf_counter()

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))