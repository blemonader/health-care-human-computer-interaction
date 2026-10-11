import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 读取原始数据
df = pd.read_csv("fast_food_consumption_health_impact_dataset.csv")

# 1. 去除 Other 性别，只保留 Male 和 Female
df = df[df["Gender"].isin(["Male", "Female"])].copy()
print(f"去除 Other 后剩余样本数: {len(df)}")

# 2. 设置随机种子，保证结果可复现
np.random.seed(42)

# 3. 重新生成 Overall_Health_Score（1-9 分）
# 设计逻辑：快餐次数越多、BMI 越高、运动越少、睡眠越少，健康评分越低
# 同时加入性别微小差异：女性平均健康评分略高 0.3 分
base_score = (
    9.0
    - 0.25 * df["Fast_Food_Meals_Per_Week"]
    - 0.08 * df["BMI"]
    + 0.30 * df["Physical_Activity_Hours_Per_Week"]
    + 0.25 * df["Sleep_Hours_Per_Day"]
    + 0.10 * df["Energy_Level_Score"]
    - 0.05 * df["Doctor_Visits_Per_Year"]
)
# 加入性别微调
base_score += np.where(df["Gender"] == "Female", 0.3, 0.0)
# 加入随机噪声，模拟真实世界的不可观测因素
base_score += np.random.normal(0, 0.8, size=len(df))
# 缩放到 1-9 并取整
df["Overall_Health_Score"] = np.clip(np.round(base_score), 1, 9).astype(int)

# 4. 重新生成 Digestive_Issues（Yes/No）
# 设计逻辑：快餐次数多、运动少、BMI 高，消化问题概率高
# 使用逻辑函数生成概率
logit = (
    0.35 * df["Fast_Food_Meals_Per_Week"]
    - 0.25 * df["Physical_Activity_Hours_Per_Week"]
    + 0.10 * df["BMI"]
    - 0.15 * df["Sleep_Hours_Per_Day"]
    - 2.0
)
# 女性消化问题概率略低
logit += np.where(df["Gender"] == "Female", -0.3, 0.0)
prob = 1 / (1 + np.exp(-logit))
df["Digestive_Issues"] = np.where(np.random.rand(len(df)) < prob, "Yes", "No")

# 5. 查看新数据的统计信息
print("\n新生成的目标变量分布：")
print("Overall_Health_Score 分布：")
print(df["Overall_Health_Score"].value_counts().sort_index())
print("\nDigestive_Issues 分布：")
print(df["Digestive_Issues"].value_counts())

# 6. 检查特征与目标的相关性
print("\n特征与 Overall_Health_Score 的相关系数：")
for col in ["Fast_Food_Meals_Per_Week", "BMI", "Physical_Activity_Hours_Per_Week",
            "Sleep_Hours_Per_Day", "Energy_Level_Score", "Doctor_Visits_Per_Year"]:
    corr = df[col].corr(df["Overall_Health_Score"])
    print(f"{col}: {corr:.3f}")

print("\n特征与 Digestive_Issues (Yes=1) 的相关系数：")
df["Digestive_Issues_num"] = (df["Digestive_Issues"] == "Yes").astype(int)
for col in ["Fast_Food_Meals_Per_Week", "BMI", "Physical_Activity_Hours_Per_Week",
            "Sleep_Hours_Per_Day"]:
    corr = df[col].corr(df["Digestive_Issues_num"])
    print(f"{col}: {corr:.3f}")

# 7. 保存为新文件，供后续训练使用
df.to_csv("fast_food_health_cleaned.csv", index=False)
print("\n已保存为 fast_food_health_cleaned.csv")