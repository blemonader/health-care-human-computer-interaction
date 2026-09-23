import streamlit as st
import pandas as pd
import joblib

# 加载你训练保存好的模型
model = joblib.load("model.pkl")

# 网页标题
st.title("Health Risk Level Prediction System")
st.write("Input vital signs to predict patient risk level")

# 侧边栏：用户输入各项生理指标
st.sidebar.header("Input Patient Vital Data")

Oxygen_Saturation = st.sidebar.slider("Oxygen_Saturation", min_value=80.0, max_value=100.0, value=94.0)
Heart_Rate = st.sidebar.slider("Heart_Rate", min_value=40.0, max_value=180.0, value=80.0)
Respiratory_Rate = st.sidebar.slider("Respiratory_Rate", min_value=8.0, max_value=30.0, value=16.0)
Temperature = st.sidebar.slider("Temperature", min_value=35.0, max_value=40.0, value=37.0)
Systolic_BP = st.sidebar.slider("Systolic_BP", min_value=60.0, max_value=180.0, value=120.0)
On_Oxygen = st.sidebar.selectbox("On_Oxygen", [0,1])
O2_Scale = st.sidebar.selectbox("O2_Scale", [1,2])
Consciousness = st.sidebar.selectbox("Consciousness", [0,1])

# 把用户输入，组装成DataFrame（和训练模型的输入格式保持一致！）
input_data = pd.DataFrame({
    "Respiratory_Rate": [Respiratory_Rate],
    "Oxygen_Saturation": [Oxygen_Saturation],
    "O2_Scale": [O2_Scale],
    "Systolic_BP": [Systolic_BP],
    "Heart_Rate": [Heart_Rate],
    "Temperature": [Temperature],
    "Consciousness": [Consciousness],
    "On_Oxygen": [On_Oxygen],
})

# 预测按钮
if st.button("Predict Risk Level"):
    pred = model.predict(input_data)
    st.subheader("Prediction Result")
    st.write(f"Predicted Risk Level: {pred[0]}")
