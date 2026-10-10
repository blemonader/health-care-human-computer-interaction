import joblib
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, balanced_accuracy_score
import sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 加载本项目保存文件
# 选用逻辑回归分类任务进行评估，保证与后续公平性改进模型一致
y_cls_test = joblib.load("y_cls_test.pkl")
y_cls_pred_lr = joblib.load("y_cls_pred_lr.pkl")
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

# 第一张图：比例指标，0~1范围
fig1, ax1 = plt.subplots(figsize=(11,6))
mf.by_group[ratio_cols].plot(ax=ax1, kind="bar", title="逻辑回归-按性别分组公平比例指标")
ax1.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig("fairness_lr_ratio.png")
print("比例指标图已保存 fairness_lr_ratio.png")

# 第二张图：样本计数指标
fig2, ax2 = plt.subplots(figsize=(11,6))
mf.by_group[count_cols].plot(ax=ax2, kind="bar", title="逻辑回归-按性别分组样本数量统计")
plt.tight_layout()
plt.savefig("fairness_lr_count.png")
print("样本数量图已保存 fairness_lr_count.png")


# 公平性指标
print("\n===== 公平性指标 =====")
eo_diff = equalized_odds_difference(
    y_cls_test,
    # 预测分类任务的模型 逻辑回归
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

X_train_scaled = joblib.load("X_train_scaled.pkl")
X_val_scaled = joblib.load("X_val_scaled.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_cls_train = joblib.load("y_cls_train.pkl")
y_cls_val = joblib.load("y_cls_val.pkl")
sens_train = joblib.load("sens_train.pkl")
sens_val = joblib.load("sens_val.pkl")


# 只使用 Gender 作为敏感属性
sens_train_gender = sens_train["Gender"]
sens_val_gender = sens_val["Gender"]
sens_test_gender = sens_test["Gender"]

# 定义约束强度候选：difference_bound 越小，公平性约束越强
bound_candidates = [0.01, 0.05, 0.10]
eps_value = 0.01

results = []

# 无约束基线
base_model = joblib.load("lr_cls_model.pkl")
y_val_pred_base = base_model.predict(X_val_scaled)

base_eo = equalized_odds_difference(
    y_cls_val, y_val_pred_base,
    sensitive_features=sens_val_gender
)
base_acc = accuracy_score(y_cls_val, y_val_pred_base)
base_f1 = f1_score(y_cls_val, y_val_pred_base, average="binary", zero_division=0)

results.append({
    "constraint": "无约束基线",
    "bound": None,
    "val_accuracy": base_acc,
    "val_f1": base_f1,
    "val_eo": base_eo,
    "model": base_model
})

print(f"\n无约束基线 - 验证集准确率: {base_acc:.4f}, F1: {base_f1:.4f}, EO差: {base_eo:.4f}")

# 遍历不同约束强度
for bound in bound_candidates:
    print(f"\n===== 训练 EqualizedOdds(difference_bound={bound}) =====")
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    constraint = EqualizedOdds(difference_bound=bound)
    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraint,
        eps=eps_value,
        max_iter=50,
    )
    mitigator.fit(
        X_train_scaled, y_cls_train,
        sensitive_features=sens_train_gender
    )

    y_val_pred = mitigator.predict(X_val_scaled)

    val_acc = accuracy_score(y_cls_val, y_val_pred)
    val_f1 = f1_score(y_cls_val, y_val_pred, average="binary", zero_division=0)
    val_bal_acc = balanced_accuracy_score(y_cls_val, y_val_pred)
    val_eo = equalized_odds_difference(
        y_cls_val, y_val_pred,
        sensitive_features=sens_val_gender
    )
    val_dp = demographic_parity_difference(
        y_cls_val, y_val_pred,
        sensitive_features=sens_val_gender
    )

    mf_val = MetricFrame(
        metrics={
            "TPR": lambda y_t, y_p: recall_score(y_t, y_p, average="binary", zero_division=0),
            "FPR": lambda y_t, y_p: 1 - recall_score(y_t, y_p, pos_label=0, zero_division=0),
            "准确率": accuracy_score,
            "预测正类比例": lambda y_t, y_p: sum(y_p == 1) / len(y_p)
        },
        y_true=y_cls_val,
        y_pred=y_val_pred,
        sensitive_features=sens_val_gender
    )

    print(f"验证集准确率: {val_acc:.4f}")
    print(f"验证集F1: {val_f1:.4f}")
    print(f"验证集平衡准确率: {val_bal_acc:.4f}")
    print(f"验证集EqualizedOdds差: {val_eo:.4f}")
    print(f"验证集DemographicParity差: {val_dp:.4f}")
    print("\n分组指标:")
    print(mf_val.by_group)

    results.append({
        "constraint": f"EqualizedOdds(bound={bound})",
        "bound": bound,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_bal_acc": val_bal_acc,
        "val_eo": val_eo,
        "val_dp": val_dp,
        "model": mitigator
    })

# 在验证集上选择最佳方案：
# 先按 EO 从小到大排序，取 EO 最小的那一档（最小 EO + 0.01 范围内）作为候选，
# 再在这些 EO 接近的候选里选 F1 最高的。
acceptable_eo_threshold = 0.10
valid = [r for r in results if r["val_eo"] <= acceptable_eo_threshold]
if valid:
    valid_sorted = sorted(valid, key=lambda x: x["val_eo"])
    min_eo = valid_sorted[0]["val_eo"]
    candidates = [r for r in valid_sorted if r["val_eo"] <= min_eo + 0.01]
    best = max(candidates, key=lambda x: x["val_f1"])
else:
    best = min(results, key=lambda x: x["val_eo"])

print(f"\n验证集选择的最佳方案: {best['constraint']}")
print(f"选择依据: EO差={best['val_eo']:.4f}, F1={best['val_f1']:.4f}")

# 在测试集上评估最佳模型
best_model = best["model"]
y_test_pred_improved = best_model.predict(X_test_scaled)

test_acc = accuracy_score(y_cls_test, y_test_pred_improved)
test_f1 = f1_score(y_cls_test, y_test_pred_improved, average="binary", zero_division=0)
test_eo = equalized_odds_difference(
    y_cls_test, y_test_pred_improved,
    sensitive_features=sens_test_gender
)
test_dp = demographic_parity_difference(
    y_cls_test, y_test_pred_improved,
    sensitive_features=sens_test_gender
)

print("\n===== 改进模型 测试集最终报告 =====")
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集F1: {test_f1:.4f}")
print(f"测试集EqualizedOdds差: {test_eo:.4f}")
print(f"测试集DemographicParity差: {test_dp:.4f}")

mf_test_improved = MetricFrame(
    metrics={
        "TPR": lambda y_t, y_p: recall_score(y_t, y_p, average="binary", zero_division=0),
        "FPR": lambda y_t, y_p: 1 - recall_score(y_t, y_p, pos_label=0, zero_division=0),
        "准确率": accuracy_score,
        "预测正类比例": lambda y_t, y_p: sum(y_p == 1) / len(y_p),
        "样本量": lambda y_t, y_p: len(y_t),
        "真实正类比例": lambda y_t, y_p: sum(y_t == 1) / len(y_t)
    },
    y_true=y_cls_test,
    y_pred=y_test_pred_improved,
    sensitive_features=sens_test_gender
)
print("\n改进模型 测试集分组指标:")
print(mf_test_improved.by_group)
print("\n改进模型 测试集组间差异:")
print(mf_test_improved.difference())


# 可视化
# 把 results 整理成 DataFrame，方便画图和对照
df_compare = pd.DataFrame([{
    "方案": r["constraint"],
    "验证集准确率": r["val_accuracy"],
    "验证集F1": r["val_f1"],
    "EqualizedOdds差": r["val_eo"],
    "DemographicParity差": r.get("val_dp", 0)
} for r in results])

print("\n===== 约束强度对比汇总（验证集） =====")
print(df_compare.to_string(index=False))

# 图：4个子图，分别展示准确率、F1、EO差、DP差随约束强度的变化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：验证集准确率
axes[0, 0].bar(df_compare["方案"], df_compare["验证集准确率"], color="steelblue")
axes[0, 0].set_title("不同约束强度下的验证集准确率")
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["验证集准确率"]):
    axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

# 子图2：验证集F1
axes[0, 1].bar(df_compare["方案"], df_compare["验证集F1"], color="darkorange")
axes[0, 1].set_title("不同约束强度下的验证集F1")
axes[0, 1].set_ylim(0, max(0.1, df_compare["验证集F1"].max() * 1.4))
axes[0, 1].set_ylabel("F1")
axes[0, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["验证集F1"]):
    axes[0, 1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

# 子图3：EqualizedOdds差
axes[1, 0].bar(df_compare["方案"], df_compare["EqualizedOdds差"], color="seagreen")
axes[1, 0].set_title("不同约束强度下的EqualizedOdds差")
axes[1, 0].set_ylabel("EO difference（越小越公平）")
axes[1, 0].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["EqualizedOdds差"]):
    axes[1, 0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

# 子图4：DemographicParity差
axes[1, 1].bar(df_compare["方案"], df_compare["DemographicParity差"], color="firebrick")
axes[1, 1].set_title("不同约束强度下的DemographicParity差")
axes[1, 1].set_ylabel("DP difference（越小越公平）")
axes[1, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["DemographicParity差"]):
    axes[1, 1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

plt.suptitle("Fairlearn约束强度对比（验证集）", fontsize=14)
plt.tight_layout()
plt.savefig("fairness_constraint_compare.png", dpi=150)
print("约束强度对比图已保存 fairness_constraint_compare.png")


# ================== 分组TPR/FPR/预测正类比例对比图 ==================
# 为每个方案按性别分组，展示各组的TPR、FPR、预测正类比例
group_records = []
for r in results:
    model = r["model"]
    y_val_pred = model.predict(X_val_scaled)
    mf_tmp = MetricFrame(
        metrics={
            "TPR": lambda y_t, y_p: recall_score(y_t, y_p, average="binary", zero_division=0),
            "FPR": lambda y_t, y_p: 1 - recall_score(y_t, y_p, pos_label=0, zero_division=0),
            "预测正类比例": lambda y_t, y_p: sum(y_p == 1) / len(y_p)
        },
        y_true=y_cls_val,
        y_pred=y_val_pred,
        sensitive_features=sens_val_gender
    )
    for group in mf_tmp.by_group.index:
        group_records.append({
            "方案": r["constraint"],
            "群体": group,
            "TPR": mf_tmp.by_group.loc[group, "TPR"],
            "FPR": mf_tmp.by_group.loc[group, "FPR"],
            "预测正类比例": mf_tmp.by_group.loc[group, "预测正类比例"]
        })

df_group = pd.DataFrame(group_records)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric in zip(axes2, ["TPR", "FPR", "预测正类比例"]):
    pivot = df_group.pivot(index="方案", columns="群体", values=metric)
    pivot.plot(ax=ax, kind="bar")
    ax.set_title(f"不同约束强度下各群体的{metric}")
    ax.set_ylim(0, max(0.2, pivot.values.max() * 1.3))
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel(metric)

plt.suptitle("Fairlearn约束强度对比：按性别分组的TPR/FPR/预测正类比例（验证集）", fontsize=13)
plt.tight_layout()
plt.savefig("fairness_constraint_group_compare.png", dpi=150)
print("分组对比图已保存 fairness_constraint_group_compare.png")