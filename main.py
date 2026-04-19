from ultralytics import YOLO

model = YOLO("yolov8s.pt")

result = model("/home/janosch/Pictures/JuliaUndIch.jpeg", conf=0.60)

result[0].show()
print(result)

