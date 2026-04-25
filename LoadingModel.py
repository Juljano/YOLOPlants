from ultralytics import YOLO

model = YOLO(R"C:\Users\jml\Projekte\Webots-Rosbot-Controller\runs\detect\train-15\weights\best.pt")

result = model(r"C:\Users\jml\Projekte\YOLOPlants\Test-images\pflanze._aug2.jpg", show=True,  conf=0.05)

result[0].show()

