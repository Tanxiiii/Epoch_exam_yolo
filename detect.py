from ultralytics import YOLO

model = YOLO('/home/tanxi/ros2/Epoch/yolo/runs/detect/yolo_competition/train_result/weights/best.pt')
#show=True代表自动弹窗显示画面
model.predict(source="0", show=True, conf=0.65)