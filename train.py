import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')

CSV_FILE_NAME = "fast_food_consumption_health_impact_dataset.csv"
# 分类目标：Digestive_Issues；回归目标：Overall_Health_Score
TARGET_CLS_COL = "Digestive_Issues"
TARGET_REG_COL = "Overall_Health_Score"
# 敏感属性
SENS_COLS = ["Gender", "Age"]
MODEL_LR_SAVE_PATH = "lr_cls_model.pkl"
MODEL_RF_SAVE_PATH = "rf_cls_model.pkl"
MODEL_REG_SAVE_PATH = "reg_model.pkl"

df = pd.read_csv(CSV_FILE_NAME)

# 编码分类标签, Yes/No → 1/0
le_cls = LabelEncoder()
df[TARGET_CLS_COL] = le_cls.fit_transform(df[TARGET_CLS_COL])

# 拆分特征，去除两个标签、两个敏感属性
X_raw = df.drop(columns=[TARGET_CLS_COL, TARGET_REG_COL] + SENS_COLS, errors="ignore")
y_cls_raw = df[TARGET_CLS_COL]
y_reg_raw = df[TARGET_REG_COL]
sensitive_raw = df[SENS_COLS]

# ========== 三分数据集 核心部分 ==========
# 第一步：切出20%作为测试集
X_temp, X_test, y_cls_temp, y_cls_test, y_reg_temp, y_reg_test, sens_temp, sens_test = train_test_split(
    X_raw, y_cls_raw, y_reg_raw, sensitive_raw,
    test_size=0.2, random_state=42, stratify=y_cls_raw
)
# 剩下80%里面再切25%作为验证集，最终：训练60%，验证20%，测试20%
X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val, sens_train, sens_val = train_test_split(
    X_temp, y_cls_temp, y_reg_temp, sens_temp,
    test_size=0.25, random_state=42, stratify=y_cls_temp
)

# 【硬性要求：scaler只在训练集fit，val/test只用transform】
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===================== 分类任务1：逻辑回归 =====================
model_lr = LogisticRegression(max_iter=500, random_state=42)
model_lr.fit(X_train_scaled, y_cls_train)

# 在验证集上评估，用于调参
y_cls_val_pred_lr = model_lr.predict(X_val_scaled)
print("===== 逻辑回归 分类任务(验证集) =====")
print("Val Accuracy: ", round(accuracy_score(y_cls_val, y_cls_val_pred_lr)*100,2),"%")
print("\nClassification Report(Val):")
print(classification_report(y_cls_val, y_cls_val_pred_lr, zero_division=0))

# 测试集最终评估
y_cls_test_pred_lr = model_lr.predict(X_test_scaled)
y_cls_test_proba_lr = model_lr.predict_proba(X_test_scaled)
print("\n===== 逻辑回归 分类任务(测试集) =====")
print("Test Accuracy: ", round(accuracy_score(y_cls_test, y_cls_test_pred_lr)*100,2),"%")
print("\nClassification Report(Test):")
print(classification_report(y_cls_test, y_cls_test_pred_lr, zero_division=0))
print("\nConfusion Matrix(Test):")
print(confusion_matrix(y_cls_test, y_cls_test_pred_lr))

# ===================== 分类任务2：随机森林 =====================
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_cls_train)

# 验证集评估
y_cls_val_pred_rf = model_rf.predict(X_val_scaled)
print("\n===== 随机森林 分类任务(验证集) =====")
print("Val Accuracy: ", round(accuracy_score(y_cls_val, y_cls_val_pred_rf)*100,2),"%")
print("\nClassification Report(Val):")
print(classification_report(y_cls_val, y_cls_val_pred_rf, zero_division=0))

# 测试集评估
y_cls_test_pred_rf = model_rf.predict(X_test_scaled)
y_cls_test_proba_rf = model_rf.predict_proba(X_test_scaled)
print("\n===== 随机森林 分类任务(测试集) =====")
print("Test Accuracy: ", round(accuracy_score(y_cls_test, y_cls_test_pred_rf)*100,2),"%")
print("\nClassification Report(Test):")
print(classification_report(y_cls_test, y_cls_test_pred_rf, zero_division=0))
print("\nConfusion Matrix(Test):")
print(confusion_matrix(y_cls_test, y_cls_test_pred_rf))

# ===================== 回归任务：Overall_Health_Score 线性回归 =====================
model_reg = LinearRegression()
model_reg.fit(X_train_scaled, y_reg_train)

# 验证集评估
y_reg_val_pred = model_reg.predict(X_val_scaled)
print("\n===== 线性回归 Overall_Health_Score回归任务(验证集) =====")
val_mae = mean_absolute_error(y_reg_val, y_reg_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_reg_val, y_reg_val_pred))
print(f"Val MAE: {val_mae:.3f}")
print(f"Val RMSE: {val_rmse:.3f}")

# 测试集最终评估
y_reg_test_pred = model_reg.predict(X_test_scaled)
test_mae = mean_absolute_error(y_reg_test, y_reg_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_test_pred))
print("\n===== 线性回归 Overall_Health_Score回归任务(测试集) =====")
print(f"Test MAE: {test_mae:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

# 保存全部需要给fair_audit.py的文件
joblib.dump(model_lr, MODEL_LR_SAVE_PATH)
joblib.dump(model_rf, MODEL_RF_SAVE_PATH)
joblib.dump(model_reg, MODEL_REG_SAVE_PATH)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_cls_test, "y_cls_test.pkl")
joblib.dump(y_reg_test, "y_reg_test.pkl")
joblib.dump(y_cls_test_pred_lr, "y_cls_pred_lr.pkl")
joblib.dump(y_cls_test_pred_rf, "y_cls_pred_rf.pkl")
joblib.dump(y_reg_test_pred, "y_reg_pred.pkl")
joblib.dump(sens_test, "sens_test.pkl")

print("\n所有模型与测试数据保存完成")
