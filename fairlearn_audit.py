import joblib
from fairlearn.metrics import MetricFrame
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, balanced_accuracy_score
import sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 加载本项目保存文件
y_cls_test = joblib.load("y_cls_test.pkl")
y_cls_pred_rf = joblib.load("y_cls_pred_rf.pkl")
sens_test = joblib.load("sens_test.pkl")
# 敏感特征选用Gender，对应数据集性别字段
sensitive_features = sens_test["Gender"]

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
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="binary", zero_division=0),
    "样本总量": count_samples,
    "真实正样本数": count_true_pos_label,
    "真实负样本数": count_true_neg_label,
    "平衡准确率": balanced_accuracy_score,
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="binary", zero_division=0),
    "TPR": lambda y_true, y_pred: recall_score(y_true, y_pred, average="binary", zero_division=0),
    "FNR": lambda y_true, y_pred: 1 - recall_score(y_true, y_pred, average="binary", zero_division=0),
    "FPR": lambda y_true, y_pred: 1 - recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    "预测正类比例": pred_positive_rate
}
mf = MetricFrame(
    metrics=metrics_dict,
    y_true=y_cls_test,
    y_pred=y_cls_pred_rf,
    sensitive_features=sensitive_features
)
print(mf.by_group)
print("\noverall metrics:")
print(mf.overall)
print("\ndifference between groups:")
print(mf.difference())

# 绘图
ratio_cols = ["accuracy","precision","平衡准确率","F1","TPR","FNR","FPR","预测正类比例"]
count_cols = ["样本总量","真实正样本数","真实负样本数"]

# 第一张图：比例指标，0~1范围
fig1, ax1 = plt.subplots(figsize=(11,6))
mf.by_group[ratio_cols].plot(ax=ax1, kind="bar", title="随机森林-按性别分组公平比例指标")
ax1.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig("fairness_rf_ratio.png")
print("比例指标图已保存 fairness_rf_ratio.png")

# 第二张图：样本计数指标
fig2, ax2 = plt.subplots(figsize=(11,6))
mf.by_group[count_cols].plot(ax=ax2, kind="bar", title="随机森林-按性别分组样本数量统计")
plt.tight_layout()
plt.savefig("fairness_rf_count.png")
print("样本数量图已保存 fairness_rf_count.png")
