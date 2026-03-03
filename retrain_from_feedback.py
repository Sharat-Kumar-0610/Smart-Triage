import pandas as pd
import joblib
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load original dataset
original_df = pd.read_csv("dataset.csv")

# Load feedback CSV
feedback_df = pd.read_csv("triage_feedback_log.csv")

# Only use VERIFIED cases
verified_df = feedback_df[feedback_df["actual_outcome"].notna()]

print("Verified samples found:", len(verified_df))

if len(verified_df) == 0:
    print("No verified feedback data found.")
    exit()

# Convert symptom JSON string back to list
verified_df["symptoms"] = verified_df["symptoms"].apply(json.loads)

# Prepare feedback dataframe
feedback_clean = pd.DataFrame({
    "symptoms": verified_df["symptoms"],
    "disease": verified_df["actual_outcome"]
})

# Merge datasets
combined_df = pd.concat([original_df, feedback_clean], ignore_index=True)

print("Total dataset size:", len(combined_df))

# Encode
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(combined_df["symptoms"])
y = combined_df["disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# Save new version
joblib.dump(model, "disease_model_v2.pkl")
joblib.dump(mlb, "symptom_encoder_v2.pkl")

print("\nNew model saved successfully.")