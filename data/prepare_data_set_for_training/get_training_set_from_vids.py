import cv2
import numpy as np
import os
from typing import Tuple

def analyze_video_and_save_yolo_data(
        video_path:str,
        roi:Tuple[int, int, int, int],
        area_threshold: int=5000,
        frame_range: int=1,
        font_scale:float=0.8,
        font_thickness:int=2,
        text_position:Tuple[int, int]=(10, 30),
        cls:int=0
        ):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    images_dir = os.path.join("train_data/images", video_name)
    labels_dir = os.path.join("train_data/labels", video_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi
    backSub = cv2.createBackgroundSubtractorMOG2()
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame_roi = frame[y:y+h, x:x+w]
        fgMask = backSub.apply(frame_roi)
        blurred = cv2.GaussianBlur(fgMask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_annotations = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                perimeter = cv2.arcLength(contour, True)
                x_min, y_min, width, height = cv2.boundingRect(contour)

                padding = 2
                if (x_min > padding and y_min > padding and
                    (x_min + width) < (w - padding) and
                    (y_min + height) < (h - padding)):

                    if len(contour) >= 5:
                        _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
                       # Convert to YOLO format using full-frame size and position
                        frame_height, frame_width = frame.shape[:2]

                        x_abs = x + x_min
                        y_abs = y + y_min

                        x_center = (x_abs + width / 2) / frame_width
                        y_center = (y_abs + height / 2) / frame_height
                        bbox_width = width / frame_width
                        bbox_height = height / frame_height

                        yolo_annotations.append(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
                        
                        cv2.drawContours(frame_roi, [contour], -1, (0, 0, 255), thickness=-1)

                        # Draw for visualization (optional)
                        cv2.rectangle(frame_roi, (x_min, y_min), (x_min+width, y_min+height), (0, 0, 255), 1)

        # Save only if there were detections
        if yolo_annotations:
            image_filename = f"{video_name}_{frame_number:04d}.png"
            label_filename = f"{video_name}_{frame_number:04d}.txt"

            cv2.imwrite(os.path.join(images_dir, image_filename), frame)
            with open(os.path.join(labels_dir, label_filename), 'w') as f:
                f.write("\n".join(yolo_annotations))

        cv2.imshow("ROI with Detections", frame_roi)
        cv2.imshow("Thresholded", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed: images in {images_dir}, labels in {labels_dir}")

# Usage of the file
roi = (103, 151, 128, 122)
analyze_video_and_save_yolo_data(
    '',  # <-- Replace with actual video path
    roi,
    area_threshold=5000,
    font_scale=0.48,
    font_thickness=1,
    text_position=(5, 16),
    cls=0
)
