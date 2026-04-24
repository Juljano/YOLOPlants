from ultralytics import YOLO

model = YOLO(R"C:\Users\jml\Projekte\Webots-Rosbot-Controller\runs\detect\train-15\weights\best.pt")

result = model(r"C:\Users\jml\Projekte\YOLOPlants\Test-images\plant-pot_aug1.jpg", show=True)

result[0].show()

