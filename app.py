import streamlit as st
import pandas as pd
import joblib
from streamlit_webrtc import webrtc_streamer
# import cv2

# 加载模型
model = joblib.load("model.pkl")

st.set_page_config(page_title="健康监测系统", layout="wide")

# 左右分栏，left_panel和main_area是左右区域自定义名
left_panel, main_area = st.columns([1, 3], border=True)

with left_panel:
    st.header("输入生命体征数据")
    Oxygen_Saturation = st.slider("血氧饱和度", min_value=80.0, max_value=100.0, value=94.0)
    Heart_Rate = st.slider("心率", min_value=40.0, max_value=180.0, value=80.0)
    Respiratory_Rate = st.slider("呼吸频率", min_value=8.0, max_value=30.0, value=16.0)
    Temperature = st.slider("体温", min_value=35.0, max_value=40.0, value=37.0)
    Systolic_BP = st.slider("收缩压", min_value=60.0, max_value=180.0, value=120.0)
    On_Oxygen = st.selectbox("是否吸氧", [0,1])
    O2_Scale = st.selectbox("供氧等级", [1,2])
    Consciousness = st.selectbox("意识状态", [0,1])
    
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

with main_area:
    # 网页标题
    st.write("输入生命体征以预测患者风险等级")
    
    # 预测按钮
    if st.button("预测风险等级"):
        pred = model.predict(input_data)
        st.subheader("预测结果")
        st.write(f"预测风险等级: {pred[0]}")
    


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



