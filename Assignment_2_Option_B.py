# Import Libraries
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Import Dataset and print info
cancer = datasets.load_breast_cancer()
print('Features: %s \n'
      'Targets: %s \n'
      % (cancer.feature_names, cancer.target_names))


# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state = 45)
### DEBUG ###
'''
print("X_train size: %s \n"
      "X_test size: %s \n"
      "y_train size: %s \n"
      "y_test size: %s \n" % (len(X_train), len(X_test), len(y_train), len(y_test)))
'''

# Create Tree Model
clf = tree.DecisionTreeClassifier()

# Define and find best GridSearch Parameters - Change verbose to 2 for details
clf_param_grid = {'criterion':('gini', 'entropy', 'log_loss'), 'splitter':('best', 'random'), 
              'max_depth':[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              'min_samples_split':[5, 7, 9, 10, 11, 12, 13, 14, 15],
              'min_samples_leaf':[2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
              'max_features':[2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf_grid = GridSearchCV(clf, clf_param_grid, cv = 5, verbose = 0)

# Find the best model parameters
clf_grid.fit(X_train, y_train)
print(clf_grid.best_params_)

# Test the Model
clf.set_params(**clf_grid.best_params_)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
### DEBUG PREDICTIONS ###
# print(y_pred)

# Compute Specificity
conf = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf.ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)

### DEBUG ###
disp = ConfusionMatrixDisplay(conf)
disp.plot()

# Print Results
print('Model Type: %s \n'
      'Model Parameters: %s \n'
      'Model Accuracy: %s \n'
      'Model Sensitivity: %s \n'
      'Model Specificity: %s \n\n'
      'Class 1 Evaluation (Malignant) \n'
      'Class 1 Precision: %s \n'
      'Class 1 Recall/Sensitivity: %s \n'
      'Class 1 F1 Score: %s \n\n'
      'Class 2 Evaluation (Benign) \n'
      'Class 2 Precision: %s \n'
      'Class 2 Recall/Sensitivity: %s \n'
      'Class 2 F1 Score: %s \n\n'
      % (type(clf).__name__, clf_grid.best_params_,
         accuracy_score(y_test, y_pred), sens, spec, 
         precision_recall_fscore_support(y_test, y_pred)[0][0],
         precision_recall_fscore_support(y_test, y_pred)[1][0],
         precision_recall_fscore_support(y_test, y_pred)[2][0],
         precision_recall_fscore_support(y_test, y_pred)[0][1],
         precision_recall_fscore_support(y_test, y_pred)[1][1],
         precision_recall_fscore_support(y_test, y_pred)[2][1]))

