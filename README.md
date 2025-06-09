# Titanic-Sink
Survivors %

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#--------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
#--------------------------------------------------------------------------

# Load the training and testing data
try:
    train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
    test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
    test_passenger_ids = test_df["PassengerId"]
except FileNotFoundError:
    print("Ensure the Titanic dataset (train.csv and test.csv) is in the correct path.")
    # As a fallback for local execution, you might need to adjust the path:
    # train_df = pd.read_csv("train.csv")
    # test_df = pd.read_csv("test.csv")
    # test_passenger_ids = test_df["PassengerId"]


def preprocess_data(df):
    """A function to preprocess the Titanic dataset."""
    # Drop columns that are less likely to be useful
    df = df.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'])

    # Fill missing 'Age' values with the median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fill missing 'Embarked' values with the mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fill missing 'Fare' in the test set
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Convert categorical 'Sex' and 'Embarked' columns to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return df

# Preprocess both training and testing data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Define features (X) and target (y)
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Split the training data for validation (optional but good practice)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_df)

#--------------------------------------------------------------------------
# 2. Model Training and Evaluation
#--------------------------------------------------------------------------

## Model 1: Logistic Regression
print("--- Logistic Regression ---")
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_val)
accuracy_log_reg = accuracy_score(y_val, y_pred_log_reg)
print(f"Validation Accuracy: {accuracy_log_reg:.4f}")

print("-" * 30)

## Model 2: Decision Tree Classifier
print("--- Decision Tree Classifier ---")
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_val)
accuracy_decision_tree = accuracy_score(y_val, y_pred_decision_tree)
print(f"Validation Accuracy: {accuracy_decision_tree:.4f}")

print("-" * 30)

## Model 3: Support Vector Machine (SVM)
print("--- Support Vector Machine ---")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_val)
accuracy_svm = accuracy_score(y_val, y_pred_svm)
print(f"Validation Accuracy: {accuracy_svm:.4f}")

#--------------------------------------------------------------------------
# 3. Prediction and Submission File Generation
#--------------------------------------------------------------------------

# You can choose any of the trained models to make predictions on the test set.
# Here, we'll use the Logistic Regression model as an example.

# Predict on the actual test data
test_predictions = log_reg.predict(X_test)

# Create the submission DataFrame
submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": test_predictions
})

# Generate the submission CSV file
submission.to_csv('submission.csv', index=False)

print("\n'submission.csv' has been created successfully!")
