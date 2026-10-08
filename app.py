import streamlit as st
import pandas as pd
import joblib
from streamlit_webrtc import webrtc_streamer
# import cv2

# 加载模型
model = joblib.load("best_model.pkl")
scaler = joblib.load ("scaler.pkl")

st.set_page_config(page_title="健康监测系统", layout="wide")

# 左右分栏，left_panel和main_area是左右区域自定义名
left_panel, main_area = st.columns([1,1], border=True)

with left_panel:
    st.header("输入生命体征数据")
    
    st.divider()
    with st.expander("使用说明与操作提示", expanded=True):
        st.write("1. 在左侧滑动条填写您的生命体征，下拉框选择吸氧、意识相关参数")
        st.write("2. 点击【预测风险等级】获得结果")
        st.write("3. 若体征录入有误，直接修改左侧滑块，再次点击预测即可重新计算")
        st.write("4. 置信度较低时，代表当前指标处于风险类别交界，需要仔细核对输入数据")
        
    Oxygen_Saturation = st.slider("血氧饱和度", min_value=80.0, max_value=100.0, value=94.0)
    Heart_Rate = st.slider("心率", min_value=40.0, max_value=180.0, value=80.0)
    Respiratory_Rate = st.slider("呼吸频率", min_value=8.0, max_value=30.0, value=16.0)
    Temperature = st.slider("体温", min_value=35.0, max_value=40.0, value=37.0)
    Systolic_BP = st.slider("收缩压", min_value=60.0, max_value=180.0, value=120.0)
    On_Oxygen_text = st.selectbox("是否吸氧", ["否","是"])
    O2_Scale_text = st.selectbox("供氧等级", ["低强度供氧","高强度供氧"])
    Consciousness_text = st.selectbox("意识状态", ["清醒","异常"])
    
    
# 把用户输入，组装成DataFrame（和训练模型的输入格式保持一致！）
On_Oxygen = 1 if On_Oxygen_text == "是" else 0
O2_Scale = 1 if O2_Scale_text == "低强度供氧" else 2
Consciousness = 1 if Consciousness_text == "异常" else 0
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

with main_area:
    # 网页标题
    st.write("输入生命体征以预测患者风险等级")
    
    # 预测按钮
    if st.button("预测风险等级"):
        raw_input = input_data.values
        input_scaled = scaler.transform(raw_input)
        pred = model.predict(input_scaled)
        pred_proba = model.predict_proba(input_scaled)
        raw_confidence = max(pred_proba[0])*100
        confidence = round(raw_confidence,2)
        st.subheader("预测结果")
        st.write(f"预测风险等级: {pred[0]}")
        st.write(f"置信度: {confidence}%")

        st.write("【结果解释】")
        if confidence < 70:
            st.warning("提示：模型置信度偏低，该组生命体征处于类别边界，结果仅供参考，建议复核体征数据。")
        elif 70 <= confidence < 90:
            st.info("模型置信度中等，可结合其他临床信息综合判断。")
        else:
            st.success("模型置信度较高，本次预测参考价值较好。")

        st.write("【适用范围说明】")
        st.write("本模型仅为课程实训演示工具，仅基于基础生命体征进行风险分级，不能替代专业医生诊断。")
        st.write("适用：基础生命体征快速初步筛查；不适用：急诊危重、多并发症患者的临床确诊。")


# if "has_face" not in st.session_state:
#     st.session_state["has_face"] = False

# # 加载opencv人脸检测器
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# def video_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#     st.session_state["has_face"] = len(faces) > 0

#     for (x,y,w_box,h_box) in faces:
#         cv2.rectangle(img,(x,y),(x+w_box,y+h_box),(0,255,0),2)
#     return frame.from_ndarray(img,format="bgr24")



# # 摄像头交互模块
# with st.expander("摄像头交互体验"):
#     webrtc_streamer(key="camera", video_frame_callback=video_callback)
#     if st.session_state["has_face"]:
#         st.success("检测到人脸，欢迎使用健康监测系统")
#     else:
#         st.info("未检测到人脸，请把脸放到摄像头画面中")



