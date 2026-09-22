# 导入
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score, accuracy_score

# sensitive_features：测试集的敏感列，注意！只用测试集，不要训练集！
sensitive_features = X_test["gender"]  

# MetricFrame：按敏感属性分组，分别计算指标
mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "recall": recall_score,   # 召回率：病人检出率，对你最重要
        "precision": precision_score
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

# 打印每组指标
print(mf.by_group)

# 整体指标
print("\noverall metrics:")
print(mf.overall)

# 查看两组之间最大差值（公平性缺口）
print("\ndifference between groups:")
print(mf.difference())
