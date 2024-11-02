# Student Performance Metric Predictor

This project aims to predict student performance by classifying students' likelihood of passing or estimating their final grades based on academic, social, and personal factors. The analysis leverages two datasets related to Math and Portuguese courses, examining attributes such as family background, study habits, and extracurricular activities to understand their impact on student outcomes.

## Dataset

The [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) dataset consists of two CSV files:
- `student-mat.csv`: Data specific to Math performance
- `student-por.csv`: Data specific to Portuguese performance

Each file includes over 30 variables related to each student’s academic and personal life, including grades for three grading periods (G1, G2, and G3), as well as features like family support, parental education, weekly study time, alcohol consumption, and much more. 

By merging the datasets, we can analyze the subset of students present in both courses, providing a comprehensive look at how different characteristics impact overall performance across subjects.

## Project Goals

1. **Classification**: To predict whether a student will "pass" or "fail" based on a threshold set for the final average grade (`G3_avg`). Passing is defined as a `G3_avg` of 10 or more.
2. **Regression**: To predict the continuous final average grade (`G3_avg`) based on student characteristics, helping estimate a student’s expected performance level.

## Methods

### Preprocessing

To prepare the data for modeling, the following preprocessing steps were taken:

1. **Data Merging**: The Math and Portuguese datasets were merged based on shared identifiers to retain only students present in both subjects.
2. **Grade Averaging**: Unified performance was calculated as the average of G1, G2, and G3 grades across subjects.
3. **Encoding Categorical Variables**:
   - Binary encoding was applied to columns with two categories, mapping values to 0 and 1.
   - One-hot encoding was used for columns with more than two categories to avoid ordinal assumptions.
4. **Standardizing Numerical Data**: Numerical columns were standardized to have a mean of 0 and standard deviation of 1 to enhance model performance and consistency.
5. **Target Variable Preparation**: Separate datasets were prepared for classification (pass/fail) and regression (final grade prediction).

### Classification Methods

For predicting whether a student will pass (G3_avg ≥ 10) or fail (G3_avg < 10), various classification models were tested. To ensure consistent results, **5-fold cross-validation** was applied across each classifier using **Stratified K-Fold** to maintain the proportion of passing and failing cases in each fold.

1. **Logistic Regression**:
   - Logistic regression is a widely used model for binary classification.
   - Each student's attributes (e.g., study time, parental education) were used as independent variables, while the binary pass/fail label served as the target variable.
   - Cross-validation accuracy scores were obtained to evaluate model performance.

2. **Support Vector Machine (SVM)**:
   - The SVM model with a **radial basis function (RBF)** kernel was trained to classify students based on their attributes.
   - Hyperparameters such as `C`, `gamma`, and `kernel type` were optimized using GridSearchCV to balance the trade-off between margin maximization and classification accuracy.

3. **Decision Tree Classifier**:
   - A decision tree classifier with a maximum depth of 2 was used to limit overfitting.
   - The model splits student data into nodes based on attribute values, providing a visual and interpretable approach for classifying student performance.
   - Cross-validation accuracy scores were calculated to gauge model reliability.

4. **Multi-Layer Perceptron (MLP) Neural Network**:
   - An MLP with a single hidden layer of 100 neurons and a **ReLU activation function** was used.
   - The model was optimized using the Adam solver, with a max iteration of 300 to ensure convergence without excessive computation.
   - MLP provided a more complex classification approach, capturing non-linear relationships between student attributes and their pass/fail outcome.

After cross-validation, the average accuracy scores across folds were calculated for each model, allowing for a performance comparison.

### Regression Methods

To predict students' final average grade (G3_avg) as a continuous outcome, several regression models were used:

1. **Linear Regression**:
   - Linear regression was applied to establish a baseline for predicting final grades.
   - MSE and R² scores were calculated across each fold to assess model performance.

2. **Support Vector Regression (SVR)**:
   - SVR with both **linear and RBF kernels** was tested, with hyperparameters optimized using grid search.
   - MSE and R² scores were computed to measure predictive accuracy.

3. **Decision Tree Regressor**:
   - Decision tree regressor was implemented with different depths to find the most accurate configuration for predicting continuous grades.
   - The model’s performance was evaluated based on MSE and R² scores.

4. **Multi-Layer Perceptron (MLP) Regressor**:
   - An MLP regressor with similar settings to the classification MLP model was used.
   - The mean squared error (MSE) and R² score provided a measure of prediction quality across folds.

### Installation

To use this project, clone the repository and install the required packages:

```bash
git clone https://github.com/dalodeju/Y3-student-performance-metric-predictor.git
cd student-performance-metric-predictor
pip install ucimlrepo scikit-learn

