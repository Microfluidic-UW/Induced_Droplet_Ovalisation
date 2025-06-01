import cv2
from pathlib import Path
import os

class VideoROISelector:
    def __init__(self, video_path):
        self.video_path = str(Path(video_path))
        self.roi = None

    def select_roi(self):
        if not os.path.exists(self.video_path):
            print(f"Error: File does not exist: {self.video_path}")
            return None

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read the first frame.")
            return None

        self.roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, w, h = self.roi
        return x, y, w, h

if __name__ == "__main__":
    video_path = "path_to_video"  # Adjust to your actual path
    roi_selector = VideoROISelector(video_path)
    roi_coords = roi_selector.select_roi()

    if roi_coords:
        print("Selected ROI coordinates:", roi_coords)
