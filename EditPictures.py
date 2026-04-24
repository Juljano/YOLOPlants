import albumentations as A
import cv2
import os

# Augmentation definieren
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
])

input_folder = r"C:\Users\jml\Desktop\images"
output_folder = "edited_images"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    image = cv2.imread(os.path.join(input_folder, file_name))
    for i in range(4):
        augmented = transform(image=image)['image']
        cv2.imwrite(os.path.join(output_folder, f"{file_name[:-4]}_aug{i}.jpg"), augmented)
