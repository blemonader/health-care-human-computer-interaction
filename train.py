import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib

CSV_FILE_NAME = "Health_Risk_Dataset.csv"
TARGET_COL = "Risk_Level"
DROP_COLS = ["Patient_ID"]
MODEL_SAVE_PATH = "best_model.pkl"

df = pd.read_csv(CSV_FILE_NAME)
df = df.drop(columns=DROP_COLS, errors="ignore")

le = LabelEncoder()
df["Consciousness"] = le.fit_transform(df["Consciousness"])

for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=[TARGET_COL], errors="ignore")
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型1：随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print("===== Random Forest 模型结果 =====")
print("Accuracy: ", round(accuracy_score(y_test, y_pred)*100,2),"%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# 模型2：逻辑回归模型
model_lr = LogisticRegression(max_iter=500, random_state=42)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_proba_lr = model_lr.predict_proba(X_test_scaled)

print("===== Logistic Regression 模型结果 =====")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_lr)*100,2),"%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))


joblib.dump(model_lr, MODEL_SAVE_PATH)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
print("\nbest_model.pkl, scaler.pkl, X_test.pkl, y_test.pkl saved successfully")


