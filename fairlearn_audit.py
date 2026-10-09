import joblib
from fairlearn.metrics import MetricFrame
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, balanced_accuracy_score
import sys
sys.stdout.reconfigure(encoding='utf-8')

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
model = joblib.load("best_model.pkl")
y_pred = model.predict(X_test)

sensitive_features = X_test["O2_Scale"]

# 由于TPR FPR等指标用于二分类，因此这里将数据转为二分类，High/Other
y_test_bin = (y_test == "High").astype(int)
y_pred_bin = (y_pred == "High").astype(int)

def count_samples(y_true, y_pred):
    return len(y_true)
def count_true_pos_label(y_true, y_pred):
    return sum(y_true == 1)
def count_true_neg_label(y_true, y_pred):
    return sum(y_true == 0)
def pred_positive_rate(y_true, y_pred):
    return sum(y_pred == 1) / len(y_pred)


metrics_dict = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="binary"),
    "样本总量": count_samples,
    "真实正样本数": count_true_pos_label,
    "真实负样本数": count_true_neg_label,
    "平衡准确率": balanced_accuracy_score,
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="binary"),
    "TPR": lambda y_true, y_pred: recall_score(y_true, y_pred, average="binary"),
    "FNR": lambda y_true, y_pred: 1 - recall_score(y_true, y_pred, average="binary"),
    "FPR": lambda y_true, y_pred: 1 - recall_score(y_true, y_pred, pos_label=0),
    "预测正类比例": pred_positive_rate
}

mf = MetricFrame(
    metrics=metrics_dict,
    y_true=y_test_bin,
    y_pred=y_pred_bin,
    sensitive_features=sensitive_features
)

print(mf.by_group)
print("\noverall metrics:")
print(mf.overall)
print("\ndifference between groups:")
print(mf.difference())
