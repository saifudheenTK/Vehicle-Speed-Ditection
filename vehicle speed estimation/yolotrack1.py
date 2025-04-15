import streamlit as st
import cv2
import tempfile
import numpy as np
from speed_estimator import SpeedEstimator
from ultralytics import YOLO
import os

st.set_page_config(layout="wide")
st.title("🚗 Vehicle Speed & Number Plate Detector (YOLOv10n + OCR)")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    model = YOLO("yolov10n.pt")
    estimator = SpeedEstimator(ppm=8, fps=fps)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to speed up
        frame = cv2.resize(frame, (640, 360))

        results = model(frame, verbose=False)[0]
        detections_raw = results.boxes.xyxy.cpu().numpy() if results.boxes else []

        annotated = estimator.estimate_speed(frame, detections_raw)
        stframe.image(annotated, channels="BGR", use_container_width=True)

    cap.release()
    st.success("✅ Done processing video.")

    if st.button("📥 Export CSV"):
        estimator.export_csv("vehicle_data.csv")
        with open("vehicle_data.csv", "rb") as f:
            st.download_button("⬇️ Download vehicle_data.csv", f, file_name="vehicle_data.csv")
