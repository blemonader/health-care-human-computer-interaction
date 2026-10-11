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


# ================== 加载数据 ==================
y_cls_test = joblib.load("y_cls_test.pkl")
y_cls_pred_lr = joblib.load("y_cls_pred_lr.pkl")
sens_test = joblib.load("sens_test.pkl")

X_train_scaled = joblib.load("X_train_scaled.pkl")
X_val_scaled = joblib.load("X_val_scaled.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_cls_train = joblib.load("y_cls_train.pkl")
y_cls_val = joblib.load("y_cls_val.pkl")
sens_train = joblib.load("sens_train.pkl")
sens_val = joblib.load("sens_val.pkl")

sens_train_gender = sens_train["Gender"]
sens_val_gender = sens_val["Gender"]
sens_test_gender = sens_test["Gender"]

# 定义约束强度候选
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

# 在验证集上选择最佳方案
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
print("best_model 类型:", type(best_model))
y_test_pred_improved = best_model.predict(X_test_scaled)

print("y_test_pred_improved 非零个数:", np.sum(y_test_pred_improved == 1))
print("y_cls_pred_lr 非零个数:", np.sum(y_cls_pred_lr == 1))
print("两者是否相同:", np.array_equal(y_test_pred_improved, y_cls_pred_lr))

# 保存改进模型预测结果，供后续文件使用
joblib.dump(y_test_pred_improved, "y_test_pred_improved.pkl")
print("改进模型预测结果已保存 y_test_pred_improved.pkl")

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


# ================== 约束强度对比可视化 ==================
df_compare = pd.DataFrame([{
    "方案": r["constraint"],
    "验证集准确率": r["val_accuracy"],
    "验证集F1": r["val_f1"],
    "EqualizedOdds差": r["val_eo"],
    "DemographicParity差": r.get("val_dp", 0)
} for r in results])

print("\n===== 约束强度对比汇总（验证集） =====")
print(df_compare.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(df_compare["方案"], df_compare["验证集准确率"], color="steelblue")
axes[0, 0].set_title("不同约束强度下的验证集准确率")
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["验证集准确率"]):
    axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

axes[0, 1].bar(df_compare["方案"], df_compare["验证集F1"], color="darkorange")
axes[0, 1].set_title("不同约束强度下的验证集F1")
axes[0, 1].set_ylim(0, max(0.1, df_compare["验证集F1"].max() * 1.4))
axes[0, 1].set_ylabel("F1")
axes[0, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["验证集F1"]):
    axes[0, 1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

axes[1, 0].bar(df_compare["方案"], df_compare["EqualizedOdds差"], color="seagreen")
axes[1, 0].set_title("不同约束强度下的EqualizedOdds差")
axes[1, 0].set_ylabel("EO difference（越小越公平）")
axes[1, 0].tick_params(axis="x", rotation=15)
for i, v in enumerate(df_compare["EqualizedOdds差"]):
    axes[1, 0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

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


# ================== Bootstrap 置信区间分析 ==================

def bootstrap_metrics(y_true, y_pred, sens, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sens = np.asarray(sens)
    n = len(y_true)

    eo_list, dp_list, f1_list, acc_list = [], [], [], []
    tpr_f_list, tpr_m_list = [], []
    fpr_f_list, fpr_m_list = [], []
    ppr_f_list, ppr_m_list = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, s = y_true[idx], y_pred[idx], sens[idx]

        if len(np.unique(yt)) < 2:
            continue

        eo_list.append(equalized_odds_difference(yt, yp, sensitive_features=s))
        dp_list.append(demographic_parity_difference(yt, yp, sensitive_features=s))
        f1_list.append(f1_score(yt, yp, average="binary", zero_division=0))
        acc_list.append(accuracy_score(yt, yp))

        for group, tpr_l, fpr_l, ppr_l in [
            ("Female", tpr_f_list, fpr_f_list, ppr_f_list),
            ("Male", tpr_m_list, fpr_m_list, ppr_m_list),
        ]:
            mask = (s == group)
            if mask.sum() < 2:
                continue
            yt_g, yp_g = yt[mask], yp[mask]
            if len(np.unique(yt_g)) < 2:
                continue
            tpr_l.append(recall_score(yt_g, yp_g, average="binary", zero_division=0))
            fpr_l.append(1 - recall_score(yt_g, yp_g, pos_label=0, zero_division=0))
            ppr_l.append(sum(yp_g == 1) / len(yp_g))

    def ci(values):
        if len(values) == 0:
            return (np.nan, np.nan)
        arr = np.array(values)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return {
        "EO差": ci(eo_list),
        "DP差": ci(dp_list),
        "F1": ci(f1_list),
        "准确率": ci(acc_list),
        "Female_TPR": ci(tpr_f_list),
        "Male_TPR": ci(tpr_m_list),
        "Female_FPR": ci(fpr_f_list),
        "Male_FPR": ci(fpr_m_list),
        "Female_预测正类比例": ci(ppr_f_list),
        "Male_预测正类比例": ci(ppr_m_list),
    }


print("\n===== Bootstrap 95%置信区间（基线模型：逻辑回归无约束） =====")
ci_base = bootstrap_metrics(y_cls_test, y_cls_pred_lr, sens_test_gender)
for k, v in ci_base.items():
    print(f"{k}: [{v[0]:.4f}, {v[1]:.4f}]")

print("\n===== Bootstrap 95%置信区间（改进模型：EqualizedOdds best） =====")
ci_improved = bootstrap_metrics(y_cls_test, y_test_pred_improved, sens_test_gender)
for k, v in ci_improved.items():
    print(f"{k}: [{v[0]:.4f}, {v[1]:.4f}]")


# ================== 基线 vs 改进 置信区间对比图 ==================
metric_keys = ["EO差", "DP差", "F1", "准确率",
               "Female_TPR", "Male_TPR",
               "Female_FPR", "Male_FPR",
               "Female_预测正类比例", "Male_预测正类比例"]

base_mid = [np.mean(ci_base[k]) if not np.isnan(ci_base[k][0]) else np.nan for k in metric_keys]
base_err_low = [ci_base[k][0] for k in metric_keys]
base_err_high = [ci_base[k][1] for k in metric_keys]
imp_mid = [np.mean(ci_improved[k]) if not np.isnan(ci_improved[k][0]) else np.nan for k in metric_keys]
imp_err_low = [ci_improved[k][0] for k in metric_keys]
imp_err_high = [ci_improved[k][1] for k in metric_keys]

x = np.arange(len(metric_keys))
width = 0.35

fig3, ax3 = plt.subplots(figsize=(16, 7))
ax3.bar(x - width/2, base_mid, width, label="基线（无约束LR）", color="steelblue",
        yerr=[np.array(base_mid) - np.array(base_err_low),
              np.array(base_err_high) - np.array(base_mid)],
        capsize=4)
ax3.bar(x + width/2, imp_mid, width, label="改进（EqualizedOdds best）", color="darkorange",
        yerr=[np.array(imp_mid) - np.array(imp_err_low),
              np.array(imp_err_high) - np.array(imp_mid)],
        capsize=4)
ax3.set_xticks(x)
ax3.set_xticklabels(metric_keys, rotation=30, ha="right")
ax3.set_ylabel("指标值（误差棒为95%置信区间）")
ax3.set_title("基线 vs 改进模型：Bootstrap 95%置信区间对比（测试集）")
ax3.legend()
plt.tight_layout()
plt.savefig("bootstrap_ci_compare.png", dpi=150)
print("Bootstrap置信区间对比图已保存 bootstrap_ci_compare.png")


# ================== EO差与DP差的Bootstrap分布直方图 ==================
def bootstrap_distribution(y_true, y_pred, sens, metric_fn, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred, sens = np.asarray(y_true), np.asarray(y_pred), np.asarray(sens)
    n = len(y_true)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, s = y_true[idx], y_pred[idx], sens[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, yp, s))
    return np.array(vals)

eo_base_dist = bootstrap_distribution(
    y_cls_test, y_cls_pred_lr, sens_test_gender,
    lambda yt, yp, s: equalized_odds_difference(yt, yp, sensitive_features=s)
)
eo_imp_dist = bootstrap_distribution(
    y_cls_test, y_test_pred_improved, sens_test_gender,
    lambda yt, yp, s: equalized_odds_difference(yt, yp, sensitive_features=s)
)
dp_base_dist = bootstrap_distribution(
    y_cls_test, y_cls_pred_lr, sens_test_gender,
    lambda yt, yp, s: demographic_parity_difference(yt, yp, sensitive_features=s)
)
dp_imp_dist = bootstrap_distribution(
    y_cls_test, y_test_pred_improved, sens_test_gender,
    lambda yt, yp, s: demographic_parity_difference(yt, yp, sensitive_features=s)
)

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
axes4[0].hist(eo_base_dist, bins=30, alpha=0.6, label="基线", color="steelblue", density=True)
axes4[0].hist(eo_imp_dist, bins=30, alpha=0.6, label="改进", color="darkorange", density=True)
axes4[0].axvline(np.mean(eo_base_dist), color="steelblue", linestyle="--")
axes4[0].axvline(np.mean(eo_imp_dist), color="darkorange", linestyle="--")
axes4[0].set_title("EO差 Bootstrap 分布")
axes4[0].set_xlabel("EO差")
axes4[0].legend()

axes4[1].hist(dp_base_dist, bins=30, alpha=0.6, label="基线", color="steelblue", density=True)
axes4[1].hist(dp_imp_dist, bins=30, alpha=0.6, label="改进", color="darkorange", density=True)
axes4[1].axvline(np.mean(dp_base_dist), color="steelblue", linestyle="--")
axes4[1].axvline(np.mean(dp_imp_dist), color="darkorange", linestyle="--")
axes4[1].set_title("DP差 Bootstrap 分布")
axes4[1].set_xlabel("DP差")
axes4[1].legend()

plt.suptitle("基线 vs 改进：EO差与DP差的Bootstrap分布（测试集）")
plt.tight_layout()
plt.savefig("bootstrap_distribution.png", dpi=150)
print("Bootstrap分布图已保存 bootstrap_distribution.png")