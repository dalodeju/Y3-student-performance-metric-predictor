# this code has StratifiedKFold

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score

# Load the datasets
math_data = pd.read_csv('student-mat.csv', sep=';')
portuguese_data = pd.read_csv('student-por.csv', sep=';')

# identifier_columns for each individual student
identifier_columns = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
]

# merge both datasets to find students who appear in both datasets, deleting the rest
combined_data = pd.merge(math_data, portuguese_data, on=identifier_columns, suffixes=('_math', '_por'))

# using the average grade from math and portuguese dataset for each grading checkpoint
combined_data['G1_avg'] = (combined_data['G1_math'] + combined_data['G1_por']) / 2
combined_data['G2_avg'] = (combined_data['G2_math'] + combined_data['G2_por']) / 2
combined_data['G3_avg'] = (combined_data['G3_math'] + combined_data['G3_por']) / 2

# columns with two categories are binary encoded (with either 0 or 1)
binary_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                  'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                  'higher', 'internet', 'romantic']

combined_data[binary_columns] = combined_data[binary_columns].apply(lambda x: x.map({'GP': 1, 'MS':0, 'yes': 1, 'no': 0, 'U': 1, 'R': 0, 'T': 1, 'A': 0, 'M': 1, 'F': 0,'LE3': 1,'GT3': 0}))

# columns with more than two categories are one-hot encoded
nominal_columns = ['Mjob', 'Fjob', 'reason', 'guardian']
combined_data = pd.get_dummies(combined_data, columns=nominal_columns, drop_first=True)

# scale the numerical columns
numerical_columns = ['age', 'absences', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']

scaler = StandardScaler()

# Fit and transform the data for numerical columns
combined_data[numerical_columns] = scaler.fit_transform(combined_data[numerical_columns])

# G3_avg is the average final math and portuguese grade

# CLASSIFICATION 

# For classification (predicting if the student will pass - pass is defined as G3_avg >= 10)
# uncomment the next 2 lines if doing classification
X_classification = combined_data.drop(columns=['G1_math', 'G1_por', 'G2_math', 'G2_por', 'G3_math', 'G3_por', 'G3_avg'])
y_classification = (combined_data['G3_avg'] >= 10).astype(int)  # 1 for pass, 0 for fail

# Divide data into training and testing data
from sklearn.model_selection import train_test_split
Xs_train, Xs_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.3, 
random_state=1, stratify=y_classification) 

## K-fold cross validation
skf= StratifiedKFold(n_splits=5)

#Logistic Regression
from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression() 
log_reg_scores= cross_val_score(log_reg,X_classification,y_classification,cv=skf, scoring='accuracy')
print(f"The classifier accuracy score of Logistic Regression is (K=5): {np.mean(log_reg_scores):.2f}")

#SVM 
from sklearn.svm import SVC 
svm_clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', 
probability=True) 
svm_scores=cross_val_score(svm_clf,X_classification,y_classification,cv=skf, scoring='accuracy')
print(f"The classifier accuracy score of SVM is (K=5): {np.mean(svm_scores):.2f}")

#Decision Tree
#Using decision tree classifier to train a model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
tree_clf = DecisionTreeClassifier(max_depth=2)
#max_depth specify the maximum depth of the tree, or how mant levels it can grow
tree_scores=cross_val_score(tree_clf,X_classification,y_classification,cv=skf, scoring='accuracy')
print(f"The classifier accuracy score of Decision Tree is (K=5): {np.mean(tree_scores):.2f}")


##Multi-layer perceptron neural network
from sklearn.neural_network import MLPClassifier
mlp_clf= MLPClassifier(hidden_layer_sizes=(100,),max_iter=300,activation='relu',solver='adam',random_state=1)
mlp_scores=cross_val_score(mlp_clf,X_classification,y_classification,cv=skf, scoring='accuracy')
print(f"The classifier accuracy score for MLP Neural Network is (K=5): {np.mean(mlp_scores):.2f}")





# REGRESSION 

# uncomment the next 2 lines if doing regression
X_regression = combined_data.drop(columns=['G1_math', 'G1_por', 'G2_math', 'G2_por', 'G3_math', 'G3_por', 'G3_avg'])
y_regression = combined_data['G3_avg']

# Impute missing values with the mean for numerical columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Define cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define custom scoring functions for MSE and R²
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# Linear Regression with K-Fold Cross-Validation
linear_model = LinearRegression()
mse_linear = cross_val_score(linear_model, X_regression, y_regression, cv=kfold, scoring=mse_scorer)
r2_linear = cross_val_score(linear_model, X_regression, y_regression, cv=kfold, scoring=r2_scorer)

# Display MSE and R² for each fold individually for Linear Regression
print(f"Linear Regression MSE for each fold: {-mse_linear}")
print(f"Linear Regression R² for each fold: {-r2_linear}")

# Display the mean MSE and R² (average MSE and R² of all folds)
print(f"Linear Regression:\nMean MSE: {-np.mean(mse_linear):.2f}\nMean R²: {np.mean(r2_linear):.2f}\n")

# Support Vector Regression with K-Fold Cross-Validation
svr_param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svr_model = GridSearchCV(SVR(), param_grid=svr_param_grid, cv=kfold, scoring=mse_scorer)
svr_model = GridSearchCV(SVR(), param_grid=svr_param_grid, cv=kfold, scoring=r2_scorer)
mse_svr = cross_val_score(svr_model, X_regression, y_regression, cv=kfold, scoring=mse_scorer)
r2_svr = cross_val_score(svr_model, X_regression, y_regression, cv=kfold, scoring=r2_scorer)

# Display MSE and R²for each fold individually for SVR
print(f"Support Vector Regression MSE for each fold: {-mse_svr}")
print(f"Support Vector Regression R² for each fold: {-r2_svr}")

# Display the mean MSE and R² (average MSE and R² of all folds)
print(f"Support Vector Regression:\nMean MSE: {-np.mean(mse_svr):.2f}\nMean R²: {np.mean(r2_svr):.2f}\n")

# Decision Tree Regression with K-Fold Cross-Validation
decision_tree_model = DecisionTreeRegressor()

# Find the best parameter to get the best MSE score
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
decision_tree_model = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=kfold, scoring=mse_scorer)
decision_tree_model = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=kfold, scoring=r2_scorer)
mse_dt = cross_val_score(decision_tree_model, X_regression, y_regression, cv=kfold, scoring=mse_scorer)
r2_dt = cross_val_score(decision_tree_model, X_regression, y_regression, cv=kfold, scoring=r2_scorer)

# Display MSE and R² for each fold individually for Decision Tree Regression 
print(f"Decision Tree Regression MSE for each fold: {-mse_dt}")
print(f"Decision Tree Regression R² for each fold: {-r2_dt}")

# Display the mean MSE and R² (average MSE abd R² of all folds)
print(f"Decision Tree Regression:\nMean MSE: {-np.mean(mse_dt):.2f}\nMean R²: {np.mean(r2_dt):.2f}\n")

# Multi-Layer Perceptron Regression with K-Fold Cross-Validation
mlp_model = MLPRegressor(max_iter=1000, learning_rate_init=0.001)
mse_mlp = cross_val_score(mlp_model, X_regression, y_regression, cv=kfold, scoring=mse_scorer)
r2_mlp = cross_val_score(mlp_model, X_regression, y_regression, cv=kfold, scoring=r2_scorer)

# Display MSE and R² for each fold individually for Linear Regression
print(f"Multi-Layer Perceptron MSE for each fold: {-mse_mlp}")
print(f"Multi-Layer Perceptron R² for each fold: {-r2_mlp}")

# Display the mean MSE and R² (average MSE and R² of all folds)
print(f"Multi-Layer Perceptron:\nMean MSE: {-np.mean(mse_mlp):.2f}\nMean R²: {np.mean(r2_mlp):.2f}\n")
