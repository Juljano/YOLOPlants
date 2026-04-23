from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data.yaml",epochs=50,imgsz=320)



model(r"C:/Users/jml/Projekte/YOLOPlants/Test-images/plant-pot.jpg", show=True)