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


├── yolotrack1.py          # Main Streamlit app
├── speed.py               # Speed estimation logic
├── yolov10n.pt            # YOLOv10n pretrained weights (place in same directory)
├── requirements.txt       # Dependencies


## 🛠️ Technologies Used

### 👁️ Computer Vision
- **YOLOv10n (Ultralytics):** Lightweight object detection model used to detect and track vehicles like cars and buses.
- **OpenCV:** Used for image processing, drawing annotations, and handling video frame operations.

### 🧠 AI/ML
- **Object Tracking:** Persistent object IDs for tracking motion across frames to estimate speed.
- **Speed Estimation Logic:** Custom Python logic based on movement across a virtual region.

### 🌐 Web Framework
- **Streamlit:** Interactive web interface for running the app in the browser. Supports file uploads, webcam input, and real-time display of annotated frames.

### 🐍 Programming & Libraries
- **Python 3.8+**
- **NumPy:** Efficient numerical operations for distance and speed calculations.
- **Ultralytics Library:** Provides easy-to-use access to YOLO models.



## 📷 Example Screenshot

![Vehicle-Speed-Detection](https://github.com/user-attachments/assets/9b3d7e5d-97d1-440b-9fcf-4425ecab5531)


