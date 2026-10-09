import streamlit as st
import pandas as pd
import joblib
from streamlit_webrtc import webrtc_streamer

# 加载模型，替换为随机森林分类模型rf_cls_model.pkl，scaler沿用训练保存的scaler.pkl
model_clf = joblib.load("rf_cls_model.pkl")
model_reg = joblib.load("reg_model.pkl")

scaler = joblib.load ("scaler.pkl")
st.set_page_config(page_title="快餐饮食与消化健康预测系统", layout="wide")
# 左右分栏，left_panel和main_area是左右区域自定义名
left_panel, main_area = st.columns([1,1], border=True)
with left_panel:
    st.header("输入用户饮食与健康数据")
    
    st.divider()
    with st.expander("使用说明与操作提示", expanded=True):
        st.write("1. 在左侧滑动条填写饮食、运动、睡眠相关参数")
        st.write("2. 点击【预测消化健康状态】获得结果")
        st.write("3. 若录入有误，直接修改左侧滑块，再次点击预测即可重新计算")
        st.write("4. 置信度较低时，代表当前指标处于类别交界，需要仔细核对输入数据")
    
    # 快餐数据集特征，和train.py训练时X_raw字段保持完全一致
    Fast_Food_Meals_Per_Week = st.slider("每周快餐就餐次数", min_value=0.0, max_value=7.0, value=2.0)
    Average_Daily_Calories = st.slider("每日平均摄入热量", min_value=1500.0, max_value=4000.0, value=2500.0)
    Physical_Activity_Hours_Per_Week = st.slider("每周运动时长(小时)", min_value=0.0, max_value=15.0, value=4.0)
    Sleep_Hours_Per_Day = st.slider("每日睡眠时长(小时)", min_value=4.0, max_value=10.0, value=7.0)
    Energy_Level_Score = st.slider("精力水平评分", min_value=0.0, max_value=10.0, value=6.0)
    Doctor_Visits_Per_Year = st.slider("每年就医次数", min_value=0.0, max_value=12.0, value=3.0)
    BMI = st.slider("BMI指数", min_value=10.0, max_value=40.0, value=25.0)

# 把用户输入，组装成DataFrame（和训练模型的输入格式保持一致！）
input_data = pd.DataFrame({
    "Fast_Food_Meals_Per_Week": [Fast_Food_Meals_Per_Week],
    "Average_Daily_Calories": [Average_Daily_Calories],
    "Physical_Activity_Hours_Per_Week": [Physical_Activity_Hours_Per_Week],
    "Sleep_Hours_Per_Day": [Sleep_Hours_Per_Day],
    "Energy_Level_Score": [Energy_Level_Score],
    "Doctor_Visits_Per_Year": [Doctor_Visits_Per_Year],
    "BMI": [BMI],
})
with main_area:
    # 网页标题
    st.write("输入饮食与健康指标，预测用户消化健康状态")
    
    # 预测按钮
    if st.button("预测消化健康状态"):
        raw_input = input_data.values
        input_scaled = scaler.transform(raw_input)
        
        pred_issues = model_clf.predict(input_scaled)
        
        pred_score = model_reg.predict(input_scaled)
        overall_score = round(pred_score[0],2)
        
        pred_proba = model_clf.predict_proba(input_scaled)
        raw_confidence = max(pred_proba[0])*100
        confidence = round(raw_confidence,2)
        
        st.subheader("预测结果")
        
        if pred_issues[0]==0:
            result_text = "无消化问题"
        else:
            result_text = "存在消化问题"
            
        st.write(f"预测消化健康状态: {result_text}")
        st.write(f"置信度: {confidence}%")
        st.write(f"总体健康指数: {overall_score}")
        
        st.write("【结果解释】")
        if confidence < 70:
            st.warning("提示：模型置信度偏低，该组指标处于类别边界，结果仅供参考，建议复核数据。")
        elif 70 <= confidence < 90:
            st.info("模型置信度中等，可结合其他健康信息综合判断。")
        else:
            st.success("模型置信度较高，本次预测参考价值较好。")
        st.write("【适用范围说明】")
        st.write("本模型仅为课程实训演示工具，基于快餐饮食习惯与基础健康指标预测消化问题，不能替代专业医生诊断。")
        st.write("适用：饮食习惯健康初步筛查；不适用：器质性消化道疾病的临床确诊。")
