import cv2
import os
import pandas as pd
import numpy as np


from libs.read_config import ReadConfig
from libs.read_and_use_model import ImageProcessor
from typing import Tuple


class ReadVideoAndAnalyse:
    """
    Class represent reading video and capturing the frames from the video
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = ReadConfig(self.config_path)
        self.config = self.config.read_config()
        self.video_path = self.config['video_path']
        self.output_folder = self.config['image_output_folder']
        self.roi = self.config['roi']
        self.area_threshold = self.config['area_treshold']
        self.vid_name = os.path.basename(self.video_path).split('.')[0]
        self.no_images_save = self.config['no._images_save']
        self.cap = None
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.particle_data = pd.DataFrame(
            columns=['Frame', 'Time (ms)', 'Area', 'Perimeter', 'Aspect Ratio', 'Circularity',
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

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y, w, h = self.roi
        frame_roi = frame[y:y + h, x:x + w]
        fgMask = self.backSub.apply(frame_roi)
        blurred = cv2.GaussianBlur(fgMask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        return frame_roi, thresh

    def analyze_contours(self, contours: list, timestamp_ms: int) -> Tuple[bool, pd.DataFrame]:
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

        # Return the particle data as a DataFrame if any data found
        if particle_data:
            return True, pd.DataFrame(particle_data)
        else:
            return False, pd.DataFrame(columns=['Frame', 'Time (ms)', 'Area', 'Perimeter', 'Aspect Ratio',
                                                'Circularity', 'Convexity', 'Elongation', 'Solidity',
                                                'Rectangularity', 'Extent', 'Max Height', 'Min Length'])

    def save_frame(self, frame: int) -> None:
        frame_filename = os.path.join(self.output_folder, f'{self.vid_name}_{self.frame_idx:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")

    def analyze_video(self) -> None:
        model = ImageProcessor(self.config['model_path'], self.config_path)
        results_AI = []  # Initialize an empty list to store DataFrames
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_roi, thresh = self.process_frame(frame)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            droplet_detected, new_data = self.analyze_contours(contours, timestamp_ms)

            # If new data is available, append it to particle_data
            if not new_data.empty:
                self.particle_data = pd.concat([self.particle_data, new_data], ignore_index=True)

            if droplet_detected:
                # Process the image and get the results
                results = model.process_image(frame)
                results_AI.append(results[1])  # Append the AI results DataFrame (results[1])

                if self.config['save_all_images'] or self.no_images_save > 0:
                    self.save_frame(frame)
                    model.save_image_AI(results[0], self.vid_name + f'{self.frame_idx:04d}.jpg')
                    if not self.config['save_all_images']:
                        self.no_images_save -= 1
            self.frame_idx += 1

            if self.config['show_video']:
                cv2.imshow('Frame', frame_roi)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Save the data to CSV after the loop
        output_folder = self.config['csv_results_files']
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the particle data to CSV
        self.particle_data.to_csv(os.path.join(output_folder, f'raw_{self.vid_name}.csv'), index=False)

        # Check if there are any AI results to save, then save to CSV
        if results_AI:  # Check if results_AI is not empty
            # Concatenate all results (which should now be DataFrames) and save
            ai_data = pd.concat(results_AI, ignore_index=True)  # This will concatenate DataFrames
            model.save_results_to_csv(ai_data, os.path.join(output_folder, f'raw_AI_{self.vid_name}.csv'))
        else:
            print("No AI results to save.")

        self.cap.release()
        cv2.destroyAllWindows()


