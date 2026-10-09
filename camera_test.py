import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import cv2
from insightface.app import FaceAnalysis

st.title("test")

app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320,320))

def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    faces = app.get(img)

    for face in faces:
        bbox = face["bbox"].astype(int)
        x1,y1,x2,y2 = bbox
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    # 绘制黑色背景条
    if len(faces) > 0:
        text = "detect face"
        color = (0,255,0)
    else:
        text = "no face"
        color = (0,0,255)
    cv2.rectangle(img, (0,0), (220,70), (0,0,0), -1)
    cv2.putText(img, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color,2)

    return frame.from_ndarray(img, format="bgr24")

streamer = webrtc_streamer(
    key="camera",
    video_frame_callback=video_callback,
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        style={"width": "100%"}
    )
)

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

