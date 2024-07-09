from ultralytics import YOLO

model = YOLO('data.yaml')

model.train(data='C:/Users/cihan/OneDrive/Masaüstü/Image/data.yaml', epochs=50, imgsz=640)
