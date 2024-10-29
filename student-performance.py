import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# For classification (predicting if the student will pass - pass is defined as G3_avg >= 10)
# uncomment the next 2 lines if doing classification
# X_classification = combined_data.drop(columns=['G1_math', 'G1_por', 'G2_math', 'G2_por', 'G3_math', 'G3_por', 'G3_avg'])
# y_classification = (combined_data['G3_avg'] >= 10).astype(int)  # 1 for pass, 0 for fail

# For regression (predicting the final grade G3_avg as a continuous variable)
# uncomment the next 2 lines if doing regression
# X_regression = combined_data.drop(columns=['G1_math', 'G1_por', 'G2_math', 'G2_por', 'G3_math', 'G3_por', 'G3_avg'])
# y_regression = combined_data['G3_avg']
