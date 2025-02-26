import cv2
import os
import pandas as pd
import numpy as np

from libs.read_config import ReadConfig
from typing import Tuple

class ReadVideoAndAnalyse:
    """
    Class represent reading video and capturing the frames from the video
    """
    
    def __init__(self, config_path:str) -> None:
        self.config_path = config_path
        self.config = ReadConfig(self.config_path)
        self.config = self.config.read_config()
        self.video_path = self.config['video_path']
        self.output_folder = self.config['image_output_folder']
        self.roi = self.config['roi']
        self.area_threshold = self.config['area_treshold']
        self.vid_name = os.path.basename(self.video_path).split('.')[0]
        self.cap = None
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.particle_data = pd.DataFrame(columns=['Frame', 'Time (ms)', 'Area', 'Perimeter', 'Aspect Ratio', 'Circularity',
                                                   'Convexity', 'Elongation', 'Solidity', 'Rectangularity', 'Extent',
                                                   'Max Height', 'Min Length'])
        self.frame_idx = 0
        self.create_output_folder()
        self.initialize_video()

    def create_output_folder(self) -> None:
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def initialize_video(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)

    def process_frame(self, frame:int) -> Tuple[int, int]:
        x, y, w, h = self.roi
        frame_roi = frame[y:y+h, x:x+w]
        fgMask = self.backSub.apply(frame_roi)
        blurred = cv2.GaussianBlur(fgMask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        return frame_roi, thresh

    def analyze_contours(self, contours:list, timestamp_ms:int) -> Tuple[bool, pd.DataFrame]:
        particle_data = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                perimeter = cv2.arcLength(contour, True)
                x_min, y_min, width, height = cv2.boundingRect(contour)
                max_height, min_length = height, width
                if len(contour) >= 5:
                    _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
                    aspect_ratio = major_axis / minor_axis
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    convexity = area / hull_area if hull_area != 0 else 0
                    elongation = major_axis / minor_axis
                    solidity = area / hull_area if hull_area != 0 else 0
                    rectangularity = area / (self.roi[2] * self.roi[3])
                    extent = area / (self.roi[2] * self.roi[3])
                    particle_data.append({
                        'filename': self.vid_name,
                        'Frame': self.cap.get(cv2.CAP_PROP_POS_FRAMES),
                        'Time (ms)': timestamp_ms,
                        'Area': area,
                        'Perimeter': perimeter,
                        'Aspect Ratio': aspect_ratio,
                        'Circularity': circularity,
                        'Convexity': convexity,
                        'Elongation': elongation,
                        'Solidity': solidity,
                        'Rectangularity': rectangularity,
                        'Extent': extent,
                        'Max Height': max_height,
                        'Min Length': min_length
                    })
                    return True, particle_data
        return False, particle_data

    def save_frame(self, frame:int) -> None:
        frame_filename = os.path.join(self.output_folder, f'{self.vid_name}_{self.frame_idx:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")

    def analyze_video(self) -> None:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_roi, thresh = self.process_frame(frame)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            droplet_detected, new_data = self.analyze_contours(contours, timestamp_ms)
            if new_data:
                self.particle_data = pd.concat([self.particle_data, pd.DataFrame(new_data)], ignore_index=True)
            if droplet_detected:
                self.save_frame(frame)
            self.frame_idx += 1
            cv2.imshow('Frame', frame_roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.particle_data.to_csv(f'data/data_to_analysis/data{self.vid_name}.csv', index=False)
        self.cap.release()
        cv2.destroyAllWindows()


