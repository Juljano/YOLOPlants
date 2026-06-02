import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:
    def __init__(self, model_path="../model/best.onnx", conf_threshold=0.01, nms_threshold=0.01):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.original_image = None

    def preprocess(self, image_path):
        """Bild laden und vorbereiten"""
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Bild nicht gefunden: {image_path}")

        self.original_image = image.copy()

        # Originalgröße speichern
        self.orig_h, self.orig_w = image.shape[:2]

        # Zielgröße
        target_size = 640

        # Skalierungsfaktor berechnen
        scale = min(
            target_size / self.orig_w,
            target_size / self.orig_h
        )

        # Neue Größe mit Seitenverhältnis
        new_w = int(self.orig_w * scale)
        new_h = int(self.orig_h * scale)

        # Bild skalieren
        image = cv2.resize(image, (new_w, new_h))

        # Schwarzes Canvas erstellen
        canvas = np.full(
            (target_size, target_size, 3),
            114,
            dtype=np.uint8
        )

        # Padding berechnen
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2

        # Bild in die Mitte setzen
        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w
        ] = image

        # Für spätere Rückskalierung speichern
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y

        image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0  # Pixelwerte normalisieren

        image = np.transpose(
            image,
            (2, 0, 1)
        )  # HWC -> CHW

        image = np.expand_dims(
            image,
            axis=0
        )  # Batch-Dimension hinzufügen

        return image
    def infer(self, image_array):
        outputs = self.session.run(None, {self.input_name: image_array})
        print(f"Inferenz abgeschlossen, Ausgabeform: {outputs[0].shape}")

        return outputs[0]

    def postprocess(self, pred):

        # (1, 6, 8400) -> (8400, 6)
        pred = pred.squeeze(0).T

        best_idx = np.argmax(pred[:, 5])

        print("Beste pot-Detection:")
        print(pred[best_idx])

        boxes = []
        scores = []
        classes = []

        for det in pred:

            x, y, w, h, plant_score, pot_score = det

            class_scores = np.array([
                plant_score,
                pot_score
            ])

            cls = int(np.argmax(class_scores))
            conf = float(np.max(class_scores))

            if conf < self.conf_threshold:
                continue

            # YOLO xywh -> xyxy
            x1 = x - w / 2
            y1 = y - h / 2

            x2 = x + w / 2
            y2 = y + h / 2

            # Letterbox entfernen
            x1 = (x1 - self.pad_x) / self.scale
            y1 = (y1 - self.pad_y) / self.scale

            x2 = (x2 - self.pad_x) / self.scale
            y2 = (y2 - self.pad_y) / self.scale

            # Begrenzen
            x1 = max(0, min(int(x1), self.orig_w))
            y1 = max(0, min(int(y1), self.orig_h))

            x2 = max(0, min(int(x2), self.orig_w))
            y2 = max(0, min(int(y2), self.orig_h))

            boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

            scores.append(conf)
            classes.append(cls)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.conf_threshold,
            self.nms_threshold
        )

        if len(indices) == 0:
            return self.original_image, set()

        class_names = {
            0: "plant",
            1: "pot"
        }

        for idx in indices.flatten():
            x, y, w, h = boxes[idx]

            conf = scores[idx]
            cls = classes[idx]

            label = class_names.get(cls, str(cls))

            cv2.rectangle(
                self.original_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                self.original_image,
                f"{label}: {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return self.original_image, set(classes)
    
    def detect(self, image_path):
        image = self.preprocess(image_path)
        pred = self.infer(image)
        result, classes = self.postprocess(pred)
        cv2.imwrite("letterbox_debug.jpg", result)
        return result, classes


