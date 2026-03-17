from ultralytics import YOLO
model = YOLO('yolo11n.pt') 
results = model.train(
    data='/home/tanxi/ros2/Epoch/yolo/my_dataset/data.yaml',  
    epochs=150,
    imgsz=640,
    device=0,
    project='yolo_competition',
    name='train_result'
)