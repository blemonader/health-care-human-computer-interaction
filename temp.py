import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CSV_FILE = "heart_attack_prediction_dataset.csv"
TARGET = "Heart Attack Risk"

df = pd.read_csv(CSV_FILE)

# 丢弃ID，不参与训练
df = df.drop("Patient ID", axis=1)

# 分离特征与标签
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# 类别特征编码
cat_cols = ["Sex","Diet","Country","Continent","Hemisphere"]
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 血压拆分：158/88拆成两个数值
def split_bp(s):
    sys, dia = s.split("/")
    return float(sys), float(dia)
X[["Systolic_BP","Diastolic_BP"]] = X["Blood Pressure"].apply(lambda x:pd.Series(split_bp(x)))
X = X.drop("Blood Pressure", axis=1)

# 划分训练测试 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 逻辑回归，开启类别平衡
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 随机森林，类别平衡
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("===== 逻辑回归 分类报告 =====")
print(classification_report(y_test, y_pred_lr))

print("\n===== 随机森林 分类报告 =====")
print(classification_report(y_test, y_pred_rf))

# 绘制简单标签分布，看正负样本数量
fig, ax = plt.subplots()
vc = y.value_counts()
ax.bar(vc.index.astype(str), vc.values)
ax.set_title("Heart Attack Risk 样本分布")
ax.set_xlabel("0=无风险，1=心脏病风险")
ax.set_ylabel("样本数量")
plt.tight_layout()
plt.show()
