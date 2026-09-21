import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

CSV_FILE_NAME = "NHANES_age_prediction.csv"
TARGET_COL = "DIQ010"
DROP_COLS = ["SEQN", "age_group"] # ID and age group column to drop
MODEL_SAVE_PATH = "model.pkl"

df = pd.read_csv(CSV_FILE_NAME)
df = df.drop(columns=DROP_COLS, errors="ignore")

# Fill missing values for numeric columns with median
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())

# Split features and target label
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("Accuracy: ", round(accuracy_score(y_test, y_pred)*100,2),"%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (for assignment 4 visualization)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save trained model for streamlit web app
joblib.dump(model, MODEL_SAVE_PATH)
print("\nmodel.pkl saved successfully")
feature_importance.to_csv("feature_importance.csv",index=False)
