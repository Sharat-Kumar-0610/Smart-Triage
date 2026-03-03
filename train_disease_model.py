import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ==========================================================
# 1️⃣ LOAD BASE DATASET
# ==========================================================
print("\nLoading dataset...")

df = pd.read_csv("d1.csv")
df["Disease"] = df["Disease"].astype(str).str.strip()

symptom_columns = [col for col in df.columns if "Symptom" in col]

# Convert symptom columns into list
df["Symptoms_List"] = df[symptom_columns].values.tolist()
df["Symptoms_List"] = df["Symptoms_List"].apply(
    lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ""]
)

# Keep only necessary columns
df = df[["Symptoms_List", "Disease"]]

original_size = len(df)
print("Original dataset size:", original_size)

# ==========================================================
# 2️⃣ TRAIN / TEST SPLIT (Before Augmentation)
# ==========================================================
X_raw = df["Symptoms_List"]
y_raw = df["Disease"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    random_state=42,
    stratify=y_raw  # keeps class distribution balanced
)

print("Training samples (before augmentation):", len(X_train_raw))
print("Testing samples:", len(X_test_raw))

# ==========================================================
# 3️⃣ AUGMENT TRAINING DATA ONLY (Prevent Leakage)
# ==========================================================
print("\nAugmenting training data...")

augmented_rows = []

for symptoms, disease in zip(X_train_raw, y_train):
    # Keep original
    augmented_rows.append((symptoms, disease))

    # Create 4 partial subsets
    for _ in range(4):
        if len(symptoms) > 1:
            subset_size = random.randint(1, len(symptoms))
            subset = random.sample(symptoms, subset_size)
        else:
            subset = symptoms
        augmented_rows.append((subset, disease))

aug_train_df = pd.DataFrame(augmented_rows, columns=["Symptoms_List", "Disease"])

print("Training size after augmentation:", len(aug_train_df))

# ==========================================================
# 4️⃣ ENCODE SYMPTOMS
# ==========================================================
mlb = MultiLabelBinarizer()

X_train = mlb.fit_transform(aug_train_df["Symptoms_List"])
X_test = mlb.transform(X_test_raw)

print("Total unique symptoms:", len(mlb.classes_))

# ==========================================================
# 5️⃣ CHECK FOR CLASS VALIDITY
# ==========================================================
unique_classes = set(aug_train_df["Disease"])

if len(unique_classes) < 2:
    raise ValueError("ERROR: Not enough disease classes to train model.")

print("Total disease classes:", len(unique_classes))

# ==========================================================
# 6️⃣ TRAIN MODEL (Balanced)
# ==========================================================
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train, aug_train_df["Disease"])

# ==========================================================
# 7️⃣ EVALUATE MODEL
# ==========================================================
print("\nEvaluating model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# 8️⃣ SAVE MODEL + ENCODER
# ==========================================================
joblib.dump(model, "disease_model.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")

print("\nModel and encoder saved successfully!")