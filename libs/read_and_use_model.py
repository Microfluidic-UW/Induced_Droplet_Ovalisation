import os
import cv2
import torch

import pandas as pd
import numpy as np

from ultralytics import YOLO
from libs.read_config import ReadConfig

class ImageProcessor:
    """
    Class represents usage of model YOLO which was trained before, and applies it to
    analyze images created in ReadVideoAndAnalyse class
    """
    def __init__(self, model_path: str, config_path: str) -> None:
        self.config_path = config_path
        self.config = ReadConfig(self.config_path)
        self.config = self.config.read_config()
        self.model = YOLO(model_path)

    def process_image(self, image: np.ndarray):
        if torch.cuda.is_available():
            if image is None:
                raise ValueError("Cannot process an empty image.")

            results = self.model(image)

            output = []
            for i, box in enumerate(results[0].boxes.xyxy):
                x_min, y_min, x_max, y_max = map(float, box[:4])
                confidence = float(results[0].boxes.conf[i])
                cls = int(results[0].boxes.cls[i])
                class_name = results[0].names[cls]
                width = x_max - x_min
                height = y_max - y_min
                aspect_ratio = width / height

                mask = results[0].masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255

                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                image = cv2.addWeighted(image, 1, mask_colored, 0.5, 0)

                output.append({
                    'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max,
                    'confidence': confidence, 'class': class_name, 'aspect_ratio': aspect_ratio
                })

                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(image, f'{class_name} {confidence:.2f}', (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_df = pd.DataFrame(output)
            return image, output_df
        else:
            raise ValueError('No CUDA available')

    def save_results_to_csv(self, results: pd.DataFrame, csv_path: str) -> None:
        if not results.empty:
            df = pd.DataFrame(results)

            # Convert any problematic columns
            for col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)

            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        else:
            print("No results to save.")

    def create_output_folder(self) -> None:
        if not os.path.exists(self.config['image_output_folder_AI']):
            os.makedirs(self.config['image_output_folder_AI'])

    def save_image_AI(self, image: np.ndarray, name: str) -> None:
        """
        Function to save the processed image to a specified path.
        """
        output_dir = os.path.dirname(self.config['image_output_folder_AI'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = name
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, image)