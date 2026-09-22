import joblib
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score, accuracy_score

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
model = joblib.load("model.pkl")
y_pred = model.predict(X_test)

sensitive_features = X_test["O2_Scale"]
metrics_dict = {
    "accuracy": accuracy_score,
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted")
}

mf = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print(mf.by_group)
print("\noverall metrics:")
print(mf.overall)
print("\ndifference between groups:")
print(mf.difference())
