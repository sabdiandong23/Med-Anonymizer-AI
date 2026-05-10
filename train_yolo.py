from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # nano模型（CPU推荐）

model.train(
    data=r"D:\Users\Desktop\作业\毕业设计\sensitive information\dataset.yaml",
    epochs=50,     # 整个训练集被模型完整学习50次
    imgsz=640,    # 输入模型的图像尺寸
    batch=4,    # 一次送进模型4张图
    device="cpu"
)
