import pandas as pd
import numpy as np

df = pd.read_csv("fast_food_consumption_health_impact_dataset.csv")
df = df[df["Gender"].isin(["Male", "Female"])].copy()
np.random.seed(42)

n = len(df)

# ================== 重新生成 Overall_Health_Score ==================
# 降低信号强度，加大噪声，让 R² 落在 0.4-0.6 之间，更真实
base_score = (
    7.5
    - 0.15 * df["Fast_Food_Meals_Per_Week"]
    - 0.06 * df["BMI"]
    + 0.20 * df["Physical_Activity_Hours_Per_Week"]
    + 0.15 * df["Sleep_Hours_Per_Day"]
    + 0.08 * df["Energy_Level_Score"]
    - 0.03 * df["Doctor_Visits_Per_Year"]
)
base_score += np.where(df["Gender"] == "Female", 0.2, 0.0)
base_score += np.random.normal(0, 1.5, size=n)   # 噪声从0.8加大到1.5
df["Overall_Health_Score"] = np.clip(np.round(base_score), 1, 9).astype(int)

# ================== 重新生成 Digestive_Issues ==================
# 关键改动：加入显式的标签翻转噪声，强制模型不可能达到100%
logit = (
    0.20 * df["Fast_Food_Meals_Per_Week"]
    - 0.15 * df["Physical_Activity_Hours_Per_Week"]
    + 0.06 * df["BMI"]
    - 0.10 * df["Sleep_Hours_Per_Day"]
    - 0.8
)
logit += np.where(df["Gender"] == "Female", -0.15, 0.0)
prob = 1 / (1 + np.exp(-logit))
labels = (np.random.rand(n) < prob).astype(int)

# 对 10% 的样本随机翻转标签，模拟真实数据里的标注错误和不可观测因素
flip_mask = np.random.rand(n) < 0.10
labels[flip_mask] = 1 - labels[flip_mask]
df["Digestive_Issues"] = np.where(labels == 1, "Yes", "No")

# 保存
df.drop(columns=["Digestive_Issues_num"], errors="ignore").to_csv(
    "fast_food_health_cleaned.csv", index=False
)

# ================== 检查 ==================
print("Digestive_Issues 分布：")
print(df["Digestive_Issues"].value_counts())
print("\nOverall_Health_Score 分布：")
print(df["Overall_Health_Score"].value_counts().sort_index())

print("\n特征与 Digestive_Issues 的相关性：")
df["di"] = (df["Digestive_Issues"] == "Yes").astype(int)
for col in ["Fast_Food_Meals_Per_Week", "Physical_Activity_Hours_Per_Week",
            "BMI", "Sleep_Hours_Per_Day"]:
    print(f"{col}: {df[col].corr(df['di']):.3f}")

print("\n特征与 Overall_Health_Score 的相关性：")
for col in ["Fast_Food_Meals_Per_Week", "Physical_Activity_Hours_Per_Week",
            "BMI", "Sleep_Hours_Per_Day", "Energy_Level_Score"]:
    print(f"{col}: {df[col].corr(df['Overall_Health_Score']):.3f}")