import joblib
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, balanced_accuracy_score
import sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 加载本项目保存文件
y_cls_test = joblib.load("y_cls_test.pkl")
y_cls_pred_lr = joblib.load("y_cls_pred_lr.pkl")
sens_test = joblib.load("sens_test.pkl")
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
    "真实正类比例": lambda y_true, y_pred: sum(y_true == 1) / len(y_true),
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="binary", zero_division=0),
    "TPR": lambda y_true, y_pred: recall_score(y_true, y_pred, average="binary", zero_division=0),
    "FNR": lambda y_true, y_pred: 1 - recall_score(y_true, y_pred, average="binary", zero_division=0),
    "FPR": lambda y_true, y_pred: 1 - recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    "预测正类比例": pred_positive_rate
}
mf = MetricFrame(
    metrics=metrics_dict,
    y_true=y_cls_test,
    y_pred=y_cls_pred_lr,
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

fig1, ax1 = plt.subplots(figsize=(11,6))
mf.by_group[ratio_cols].plot(ax=ax1, kind="bar", title="逻辑回归-按性别分组公平比例指标")
ax1.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig("fairness_lr_ratio.png")
print("比例指标图已保存 fairness_lr_ratio.png")

fig2, ax2 = plt.subplots(figsize=(11,6))
mf.by_group[count_cols].plot(ax=ax2, kind="bar", title="逻辑回归-按性别分组样本数量统计")
plt.tight_layout()
plt.savefig("fairness_lr_count.png")
print("样本数量图已保存 fairness_lr_count.png")


# 公平性指标
print("\n===== 公平性指标 =====")
eo_diff = equalized_odds_difference(
    y_cls_test,
    y_cls_pred_lr,
    sensitive_features=sensitive_features
)
dp_diff = demographic_parity_difference(
    y_cls_test,
    y_cls_pred_lr,
    sensitive_features=sensitive_features
)

print("\nequalized_odds_difference:")
print(eo_diff)
print("\ndemographic_parity_difference:")
print(dp_diff)


# ================== 性别 × 年龄段 交叉分析 ==================

def age_to_group(age):
    if age < 35:
        return "Young"
    elif age <= 50:
        return "Middle"
    else:
        return "Old"

sens_test_age_group = sens_test["Age"].apply(age_to_group)
sens_test_cross = sens_test["Gender"].astype(str) + "_" + sens_test_age_group.astype(str)

print("\n===== 性别 × 年龄段 组合分布（测试集） =====")
print(sens_test_cross.value_counts())


print("\n===== 交叉分析：基线模型（逻辑回归无约束） =====")

mf_cross_base = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, average="binary", zero_division=0),
        "TPR": lambda y_t, y_p: recall_score(y_t, y_p, average="binary", zero_division=0),
        "FPR": lambda y_t, y_p: 1 - recall_score(y_t, y_p, pos_label=0, zero_division=0),
        "预测正类比例": lambda y_t, y_p: sum(y_p == 1) / len(y_p),
        "样本量": lambda y_t, y_p: len(y_t),
        "真实正类比例": lambda y_t, y_p: sum(y_t == 1) / len(y_t),
    },
    y_true=y_cls_test,
    y_pred=y_cls_pred_lr,
    sensitive_features=sens_test_cross
)

print("\n分组指标：")
print(mf_cross_base.by_group)
print("\n组间差异：")
print(mf_cross_base.difference())

eo_cross_base = equalized_odds_difference(
    y_cls_test,
    y_cls_pred_lr,
    sensitive_features=sens_test_cross
)
dp_cross_base = demographic_parity_difference(
    y_cls_test,
    y_cls_pred_lr,
    sensitive_features=sens_test_cross
)
print(f"\n交叉EqualizedOdds差: {eo_cross_base:.4f}")
print(f"交叉DemographicParity差: {dp_cross_base:.4f}")


# 交叉分析可视化
fig_cross, axes_cross = plt.subplots(1, 2, figsize=(16, 6))

# 子图1：各交叉组的TPR/FPR
mf_cross_base.by_group[["TPR", "FPR"]].plot(
    ax=axes_cross[0], kind="bar", title="基线模型：性别×年龄组的TPR/FPR"
)
axes_cross[0].set_ylim(0, 1.05)
axes_cross[0].tick_params(axis="x", rotation=30)

# 子图2：各交叉组的预测正类比例与样本量
mf_cross_base.by_group[["预测正类比例", "真实正类比例"]].plot(
    ax=axes_cross[1], kind="bar", title="基线模型：性别×年龄组的预测与真实正类比例"
)
axes_cross[1].set_ylim(0, 1.05)
axes_cross[1].tick_params(axis="x", rotation=30)

plt.suptitle("性别 × 年龄段 交叉公平性分析（基线模型，测试集）", fontsize=13)
plt.tight_layout()
plt.savefig("fairness_cross_gender_age.png", dpi=150)
print("交叉分析图已保存 fairness_cross_gender_age.png")


# 交叉分析样本量可视化
fig_cross_count, ax_cross_count = plt.subplots(figsize=(10, 5))
mf_cross_base.by_group["样本量"].plot(ax=ax_cross_count, kind="bar", color="steelblue")
ax_cross_count.set_title("性别 × 年龄段 各交叉组样本量（测试集）")
ax_cross_count.set_ylabel("样本量")
ax_cross_count.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("fairness_cross_sample_count.png", dpi=150)
print("交叉分析样本量图已保存 fairness_cross_sample_count.png")