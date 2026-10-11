import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ================== 读取数据 ==================
df = pd.read_csv("fast_food_health_cleaned.csv")

TARGET_CLS = "Digestive_Issues"
TARGET_REG = "Overall_Health_Score"

le = LabelEncoder()
df[TARGET_CLS] = le.fit_transform(df[TARGET_CLS])   # Yes->1, No->0

# 去掉可能存在的辅助列
df = df.drop(columns=["Digestive_Issues_num"], errors="ignore")

# 敏感属性单独保存
sens = df[["Gender", "Age"]].copy()

# 特征：去掉两个目标列和 Gender（Gender 只用于公平性审计）
X = df.drop(columns=[TARGET_CLS, TARGET_REG, "Gender"])

# 强制只保留数值列，防止残留字符串
X = X.select_dtypes(include=[np.number])

y_cls = df[TARGET_CLS]
y_reg = df[TARGET_REG]

print("特征列：", list(X.columns))
print("特征形状：", X.shape)

# ================== 划分数据集 ==================
X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te, s_tr, s_te = train_test_split(
    X, y_cls, y_reg, sens, test_size=0.2, random_state=42, stratify=y_cls
)

# 标准化
scaler = StandardScaler()
X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

print("\n" + "=" * 60)
print("分类任务：预测 Digestive_Issues")
print("=" * 60)

# 逻辑回归
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_tr, yc_tr)
pred_lr = lr.predict(X_te)
print("\n--- 逻辑回归 ---")
print("Accuracy :", round(accuracy_score(yc_te, pred_lr), 4))
print("Precision:", round(precision_score(yc_te, pred_lr, zero_division=0), 4))
print("Recall   :", round(recall_score(yc_te, pred_lr, zero_division=0), 4))
print("F1       :", round(f1_score(yc_te, pred_lr, zero_division=0), 4))
print("混淆矩阵 :\n", confusion_matrix(yc_te, pred_lr))
print(classification_report(yc_te, pred_lr, zero_division=0))

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, yc_tr)
pred_rf = rf.predict(X_te)
print("\n--- 随机森林 ---")
print("Accuracy :", round(accuracy_score(yc_te, pred_rf), 4))
print("Precision:", round(precision_score(yc_te, pred_rf, zero_division=0), 4))
print("Recall   :", round(recall_score(yc_te, pred_rf, zero_division=0), 4))
print("F1       :", round(f1_score(yc_te, pred_rf, zero_division=0), 4))
print("混淆矩阵 :\n", confusion_matrix(yc_te, pred_rf))
print(classification_report(yc_te, pred_rf, zero_division=0))

print("\n" + "=" * 60)
print("回归任务：预测 Overall_Health_Score")
print("=" * 60)

# 线性回归
lin = LinearRegression()
lin.fit(X_tr, yr_tr)
pred_lin = lin.predict(X_te)
print("\n--- 线性回归 ---")
print("MAE :", round(mean_absolute_error(yr_te, pred_lin), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(yr_te, pred_lin)), 4))
print("R2  :", round(r2_score(yr_te, pred_lin), 4))

# 随机森林回归
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_tr, yr_tr)
pred_rfr = rfr.predict(X_te)
print("\n--- 随机森林回归 ---")
print("MAE :", round(mean_absolute_error(yr_te, pred_rfr), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(yr_te, pred_rfr)), 4))
print("R2  :", round(r2_score(yr_te, pred_rfr), 4))

# 按性别分组的召回率
print("\n" + "=" * 60)
print("分类任务按性别分组的召回率（逻辑回归）")
print("=" * 60)
for g in s_te["Gender"].unique():
    mask = (s_te["Gender"] == g).values
    if mask.sum() > 0:
        rec = recall_score(yc_te[mask], pred_lr[mask], zero_division=0)
        acc = accuracy_score(yc_te[mask], pred_lr[mask])
        print(f"{g}: 样本数={mask.sum()}, Accuracy={acc:.4f}, Recall={rec:.4f}")

# 按性别分组的回归 MAE
print("\n" + "=" * 60)
print("回归任务按性别分组的 MAE（线性回归）")
print("=" * 60)
for g in s_te["Gender"].unique():
    mask = (s_te["Gender"] == g).values
    if mask.sum() > 0:
        mae = mean_absolute_error(yr_te[mask], pred_lin[mask])
        print(f"{g}: 样本数={mask.sum()}, MAE={mae:.4f}")