# fairness_audit.py Assignment 2 Fairness audit
import pandas as pd
import joblib
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

CSV_FILE_NAME = "NHANES_age_prediction.csv"
TARGET_COL = "DIQ010"
DROP_COLS = ["SEQN", "age_group"]

# load dataset
df = pd.read_csv(CSV_FILE_NAME)
df = df.drop(columns=DROP_COLS, errors="ignore")

# fill missing values for numeric columns with median
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())

# split features and target label
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# same split parameters as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# load saved model
model = joblib.load("model.pkl")
y_pred = model.predict(X_test)

# protected attribute: RIAGENDR (gender)
sensitive_feature = X_test["RIAGENDR"]

# fairness evaluation, calculate recall grouped by gender
metric_frame = MetricFrame(
    metrics=recall_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature,
    sample_params={"zero_division": 0}
)

print("==== Fairlearn Fairness Evaluation (Recall grouped by gender) ====")
print(metric_frame.by_group)
print("\nOverall recall: ", metric_frame.overall)
print("Recall difference between gender groups: ", metric_frame.difference())
