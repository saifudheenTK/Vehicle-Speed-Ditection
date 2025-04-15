import cv2
import numpy as np
import pytesseract
import csv
from norfair import Detection, Tracker

class SpeedEstimator:
    def __init__(self, ppm=8, fps=30):
        self.tracker = Tracker(distance_function=self.euclidean_distance, distance_threshold=30)
        self.previous_positions = {}
        self.vehicle_speeds = {}
        self.plate_data = []
        self.ppm = ppm
        self.fps = fps
        self.frame_count = {}

        # Set path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def euclidean_distance(self, det1, det2):
        p1 = det1.estimate if hasattr(det1, 'estimate') else det1.points
        p2 = det2.estimate if hasattr(det2, 'estimate') else det2.points
        return np.linalg.norm(p1 - p2)

    def estimate_speed(self, frame, detections_raw):
        detections = []
        for box in detections_raw:
            x1, y1, x2, y2 = box.astype(int)
            center = np.array([[int((x1 + x2) / 2), int((y1 + y2) / 2)]])
            detections.append(Detection(points=center, data={"bbox": [x1, y1, x2, y2]}))

        tracked_objects = self.tracker.update(detections=detections)

        for obj in tracked_objects:
            obj_id = obj.id
            cx, cy = obj.estimate[0]
            bbox = obj.last_detection.data["bbox"]
            x1, y1, x2, y2 = bbox

            # Count frames per object
            if obj_id not in self.frame_count:
                self.frame_count[obj_id] = 1
                self.previous_positions[obj_id] = (cx, cy)
            else:
                self.frame_count[obj_id] += 1
                prev_pos = self.previous_positions[obj_id]
                distance_px = np.linalg.norm(np.array([cx, cy]) - np.array(prev_pos))
                distance_m = distance_px / self.ppm
                time_s = self.frame_count[obj_id] / self.fps
                speed_mps = distance_m / time_s if time_s > 0 else 0
                speed_kmph = speed_mps * 3.6
                self.vehicle_speeds[obj_id] = speed_kmph
                self.previous_positions[obj_id] = (cx, cy)

            # Number plate extraction
            roi = frame[y1:y2, x1:x2]
            plate = pytesseract.image_to_string(roi, config='--psm 8')
            plate = ''.join(filter(str.isalnum, plate)).upper()

            # Store info
            speed_value = round(self.vehicle_speeds.get(obj_id, 0), 2)
            vehicle_info = {
                "id": obj_id,
                "speed": speed_value,
                "plate": plate if len(plate) >= 5 else "UNKNOWN"
            }
            self.plate_data.append(vehicle_info)

            # Draw on frame
            label = f"ID:{obj_id} | Speed: {speed_value:.1f} km/h | Plate: {vehicle_info['plate']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return frame

    def export_csv(self, filename="vehicle_data.csv"):
        if self.plate_data:
            with open(filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["id", "plate", "speed"])
                writer.writeheader()
                for row in self.plate_data:
                    writer.writerow(row)
