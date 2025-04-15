import cv2
import numpy as np
import pytesseract
import math
from time import time
from norfair import Detection, Tracker
import re

# Set Tesseract path (update based on your OS)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # <-- Change if needed

class SpeedEstimator:
    def __init__(self, model):
        self.model = model
        self.tracker = Tracker(distance_function=self.euclidean_distance, distance_threshold=30)
        self.prev_positions = {}
        self.prev_time = {}
        self.records = []
        self.pixels_per_meter = 8  # Customize based on your camera setup

    def euclidean_distance(self, d1, d2):
        return np.linalg.norm(d1.points - d2.points)

    def convert_bbox_to_detection(self, bbox, score):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return Detection(np.array([cx, cy]), scores=np.array([score]))

    def estimate_speed(self, frame):
        results = self.model.predict(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 2 or cls == 7:  # car or truck
                detection = self.convert_bbox_to_detection(box.xyxy[0].tolist(), box.conf[0])
                detection.data["bbox"] = box.xyxy[0].tolist()
                detections.append(detection)

        tracked_objects = self.tracker.update(detections)

        for obj in tracked_objects:
            t_id = obj.id
            cx, cy = obj.estimate[0]
            bbox = obj.last_detection.data["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            if t_id in self.prev_positions:
                prev_cx, prev_cy = self.prev_positions[t_id]
                dx = cx - prev_cx
                dy = cy - prev_cy
                distance = math.sqrt(dx ** 2 + dy ** 2)

                curr_time = time()
                time_elapsed = curr_time - self.prev_time[t_id]
                if time_elapsed > 0:
                    speed_mps = (distance / self.pixels_per_meter) / time_elapsed
                    speed_kmph = speed_mps * 3.6
                    speed_text = f"{speed_kmph:.2f} km/h"

                    # Number plate extraction
                    plate_img = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    plate_text = pytesseract.image_to_string(gray, config='--psm 7')
                    plate_text = re.sub(r'\W+', '', plate_text)  # clean up

                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {t_id} | {speed_text}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Plate: {plate_text}", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Save record
                    self.records.append({
                        "vehicle_id": t_id,
                        "speed_kmph": round(speed_kmph, 2),
                        "number_plate": plate_text,
                        "timestamp": curr_time
                    })

            self.prev_positions[t_id] = (cx, cy)
            self.prev_time[t_id] = time()

        return frame
