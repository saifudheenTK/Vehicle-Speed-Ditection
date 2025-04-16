# 🚗 Vehicle Speed Detection using YOLOv10 and Streamlit

This project is a real-time vehicle speed detection web application built with **YOLOv10**, **OpenCV**, and **Streamlit**. It can process both uploaded video files and webcam feeds to track vehicles and estimate their speed based on their motion across a defined region.

## 🔍 Features

- 🎥 Upload video or use your webcam (locally)
- 🚘 Detect and track cars and buses using **YOLOv10n**
- ⚡ Estimate and display the speed of each tracked vehicle
- 📐 Customizable detection region for speed calculation
- 💻 Simple and interactive web UI using **Streamlit**

---

## 📁 Project Structure

```bash
├── yolotrack1.py          # Main Streamlit app
├── speed.py               # Speed estimation logic
├── yolov10n.pt            # YOLOv10n pretrained weights (place in same directory)
├── requirements.txt       # Dependencies


📸 Input Modes
Video Upload: Choose any .mp4, .avi, or .mov file.

Webcam: Works only in local environments. Click the “Start Webcam” button.

⚙️ How Speed Estimation Works
Detects vehicles using YOLOv10

Tracks their position frame by frame

Calculates speed when they cross a virtual line

Displays speed on the frame (in pixels/second converted to km/h)

📌 Dependencies
streamlit

opencv-python

ultralytics

numpy
