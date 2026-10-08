# ============================================================
# DecodeLabs - Artificial Intelligence Project 2
# Data Classification Using AI
# ============================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# 1. PROJECT TITLE
# ------------------------------------------------------------

print("=" * 60)
print("        DECODELABS - ARTIFICIAL INTELLIGENCE")
print("             PROJECT 2: DATA CLASSIFICATION")
print("=" * 60)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

dataset_path = "iris.csv"

if not os.path.exists(dataset_path):
    print("\nERROR: Dataset file not found!")
    print("Please make sure iris.csv is inside the dataset folder.")
    exit()

data = pd.read_csv(dataset_path)

print("\n[1] DATASET LOADED SUCCESSFULLY")
print("-" * 60)

print("Number of rows    :", data.shape[0])
print("Number of columns :", data.shape[1])


# ------------------------------------------------------------
# 3. UNDERSTAND THE DATASET
# ------------------------------------------------------------

print("\n[2] FIRST FIVE RECORDS")
print("-" * 60)

print(data.head())


print("\n[3] DATASET INFORMATION")
print("-" * 60)

print(data.info())


print("\n[4] MISSING VALUES")
print("-" * 60)

print(data.isnull().sum())


print("\n[5] CLASS DISTRIBUTION")
print("-" * 60)

print(data["species"].value_counts())


# ------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = data.drop("species", axis=1)
y = data["species"]

print("\n[6] FEATURES AND TARGET")
print("-" * 60)

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print("species")


# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n[7] TRAINING AND TESTING DATA")
print("-" * 60)

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ------------------------------------------------------------
# 6. CREATE CLASSIFICATION MODEL
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    random_state=42
)


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\n[8] MODEL TRAINING")
print("-" * 60)

print("Decision Tree Classifier trained successfully!")


# ------------------------------------------------------------
# 8. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\n[9] MODEL PREDICTIONS")
print("-" * 60)

print("Actual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)


# ------------------------------------------------------------
# 9. CALCULATE ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n[10] MODEL ACCURACY")
print("-" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------------------
# 10. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n[11] CLASSIFICATION REPORT")
print("-" * 60)

print(classification_report(y_test, y_pred))


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

print("\n[12] CONFUSION MATRIX")
print("-" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ------------------------------------------------------------
# 12. TEST WITH NEW DATA
# ------------------------------------------------------------

print("\n[13] TESTING WITH NEW DATA")
print("-" * 60)

new_flower = [[
    5.1,   # sepal length
    3.5,   # sepal width
    1.4,   # petal length
    0.2    # petal width
]]

prediction = model.predict(new_flower)

print("Input measurements:")
print(new_flower)

print("\nPredicted species:")
print(prediction[0])


# ------------------------------------------------------------
# 13. FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("                 PROJECT COMPLETED")
print("=" * 60)

print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Algorithm: Decision Tree")
print("Dataset: Iris Dataset")

print("\nThe model was successfully trained and tested.")
print("=" * 60)