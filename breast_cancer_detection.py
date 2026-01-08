# ---------------------------------
# Breast Cancer Detection (FINAL)
# ---------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------
# Load dataset
# ---------------------------------
data = pd.read_csv("breast_cancer.csv")

# ---------------------------------
# Drop useless columns
# ---------------------------------
# Drop ID column if present
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)

# Drop unnamed empty column
data.drop(columns=['Unnamed: 32'], errors='ignore', inplace=True)

# ---------------------------------
# Encode target
# ---------------------------------
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# ---------------------------------
# Keep ONLY numeric columns
# ---------------------------------
data = data.select_dtypes(include=[np.number])

# ---------------------------------
# Separate X and y
# ---------------------------------
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# ---------------------------------
# Handle missing values (STRONG)
# ---------------------------------
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# ---------------------------------
# Train-test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# Logistic Regression (SAFE SOLVER)
# ---------------------------------
lr = LogisticRegression(max_iter=1000, solver='liblinear')
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ---------------------------------
# SVM
# ---------------------------------
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# ---------------------------------
# Random Forest
# ---------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ---------------------------------
# Evaluation
# ---------------------------------
print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\nSVM")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))