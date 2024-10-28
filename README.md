# Student Performance Metric Predictor

This project aims to predict student performance by classifying students' likelihood of passing or estimating their final grades based on academic, social, and personal factors. The analysis leverages two datasets related to Math and Portuguese courses, examining attributes such as family background, study habits, and extracurricular activities to understand their impact on student outcomes.

## Dataset

The data is derived from two CSV files:
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

## Usage

### Classification

To classify whether a student is likely to pass, use the classification model. This model takes various student attributes (e.g., study time, parental education) as input and predicts a binary outcome:
- 1 for "Pass" (G3_avg ≥ 10)
- 0 for "Fail" (G3_avg < 10)

### Regression

To predict a student's final average grade as a continuous variable, use the regression model. The model predicts the `G3_avg` based on student features, providing an estimated grade level.

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/dalodeju/Y3-student-performance-metric-predictor.git
cd student-performance-metric-predictor
pip install ucimlrepo
pip install scikit-learn
