import os
import cv2

import pandas as pd
import numpy as np

from ultralytics import YOLO
from libs.read_config import ReadConfig

class ImageProcessor:
    """
    Class represents usage of model YOLO which was trained before, and aply it to
    analyse images created in ReadVideoAndAnalyse class
    """
    def __init__(self, model_path:YOLO, config_path:str) -> None:
        self.config_path = config_path
        self.config = ReadConfig(self.config_path)
        self.config = self.config.read_config()
        self.model = YOLO(model_path)

    def process_image(self):
        image = cv2.imread(self.config['image_output_folder'])
        if image is None:
            raise ValueError(f"Cannot load image: {self.config['image_output_folder']}")

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
                'file_name': os.path.basename(self.config['image_output_folder']),
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': confidence,
                'class': class_name,
                'aspect_ratio': aspect_ratio
            })

            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, f'{class_name} {confidence:.2f}', (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not os.path.exists(self.config['image_output_folder_AI']):
            os.makedirs(self.config['image_output_folder_AI'])
        output_image_path = os.path.join(self.config['image_output_folder_AI'], os.path.basename(self.config['image_output_folder']))
        cv2.imwrite(output_image_path, image)

        return output

    def save_results_to_csv(self, results, csv_path):
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")