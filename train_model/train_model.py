from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    model = YOLO('yolo11s.pt')
    print(model.info())

    results = model.train(data='path_to_yaml_file.yaml', epochs=100, imgsz=640)
    

