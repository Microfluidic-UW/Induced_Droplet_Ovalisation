import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

model = YOLO('yolo11s_05_2025.pt')


def yolo_predict(image_bgr, aspect_ratio):
    results = model(image_bgr)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls_id]

        detections.append({
            'class': label,
            'confidence': confidence,
            'x_min': x1,
            'y_min': y1,
            'x_max': x2,
            'y_max': y2,
            'aspect_ratio': aspect_ratio
        })

    return detections


def analyze_video_and_process_data(video_path, output_folder, roi, yolo_model,
                                   area_threshold=2000, font_scale=0.8, font_thickness=2, text_position=(10, 30)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi  # ROI coordinates (x, y, width, height)
    backSub = cv2.createBackgroundSubtractorMOG2()
    detections_csv = []
    frame_number = 0

    def draw_dashed_rectangle(image, top_left, bottom_right, color, thickness, dash_length):
        x1, y1 = top_left
        x2, y2 = bottom_right
        # Draw horizontal lines
        for x_pos in range(x1, x2, dash_length * 2):
            cv2.line(image, (x_pos, y1), (min(x_pos + dash_length, x2), y1), color, thickness)
            cv2.line(image, (x_pos, y2), (min(x_pos + dash_length, x2), y2), color, thickness)
        # Draw vertical lines
        for y_pos in range(y1, y2, dash_length * 2):
            cv2.line(image, (x1, y_pos), (x1, min(y_pos + dash_length, y2)), color, thickness)
            cv2.line(image, (x2, y_pos), (x2, min(y_pos + dash_length, y2)), color, thickness)
    
    no_of_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_roi = frame[y:y + h, x:x + w]

        fgMask = backSub.apply(frame_roi)
        blurred = cv2.GaussianBlur(fgMask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
                        aspect_ratio = major_axis / minor_axis
                        if perimeter != 0:
                            circularity = 4 * np.pi * area / (perimeter ** 2)
                            if area > 50 and 0.5 < circularity < 1.0:
                                # Create a copy of the original frame to draw on
                                frame_with_mask = frame.copy()

                                # Shift contour points to full-frame coordinates
                                offset_contour = contour + (x, y)  # ROI offset

                                # Draw red mask on the original frame
                                cv2.drawContours(frame_with_mask, [offset_contour], -1, (0, 0, 255),
                                                 thickness=cv2.FILLED)

                                # Run YOLO model on the ROI
                                cv2.imshow('frame', frame_with_mask)
                                detections = yolo_model(frame_with_mask, aspect_ratio)
                                filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
                                if detections:
                                    boxes = [(d['x_min'], d['y_min'], d['x_max'] - d['x_min'], d['y_max'] - d['y_min'])
                                             for d in detections]
                                    scores = [d['confidence'] for d in detections]
                                    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)

                                    save_frame_flag = True
                                    y_offset = 0

                                    for i in indices:
                                        i = int(i)  # Handles both flat and nested list cases safely
                                        det = detections[i]
                                        label = det['class']
                                        conf = det['confidence']
                                        x1, y1, x2, y2 = det['x_min'], det['y_min'], det['x_max'], det['y_max']

                                        # Add detection to CSV data
                                        detections_csv.append({
                                            'frame_number': frame_number,
                                            'timestamp_ms': timestamp_ms,
                                            'class': label,
                                            'confidence': conf,
                                            'x_min': x1,
                                            'y_min': y1,
                                            'x_max': x2,
                                            'y_max': y2,
                                            'aspect_ratio': det['aspect_ratio']
                                        })

                                        # Draw detection
                                        cv2.rectangle(frame_with_mask, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                        label_text = f"{label} {conf:.2f}"
                                        cv2.putText(frame_with_mask, label_text, (x1, y1 - 5 - y_offset),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        y_offset += 20

                                        if label == 'proteolytic':
                                            # save_frame_flag = True
                                            cv2.imwrite(os.path.join(output_folder, f"positive_{frame_number:04d}.png"), frame_with_mask)

                                    if save_frame_flag:
                                        if no_of_frames < 40:
                                            no_of_frames += 1
                                            cv2.imwrite(filename, frame_with_mask)

        # Draw ROI and frame info on the display frame (not saved)
        draw_dashed_rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1, 10)
        text = f"Frame: {frame_number} | Time: {int(timestamp_ms)} ms"
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness)

        cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detections_csv:  # Only save CSV if there are detections
        csv_path = os.path.join(output_folder, "detections.csv")
        pd.DataFrame(detections_csv).to_csv(csv_path, index=False)
        print(f"[✓] Detection data saved to {csv_path}")


# Configuration
roi = (70, 50, 170, 120)  # (x, y, width, height)
output_folder = "detections_from_video"

# Process video
analyze_video_and_process_data(
    'video_path', # path to video. Change it :)
    output_folder,
    roi,
    yolo_predict,
    font_scale=0.48,
    font_thickness=1,
    text_position=(5, 16)
)
