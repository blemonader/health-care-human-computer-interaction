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
