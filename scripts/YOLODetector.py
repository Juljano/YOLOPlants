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
        
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def infer(self, image_array):
        """Führe Inferenz durch"""
        outputs = self.session.run(None, {self.input_name: image_array})
        return outputs[0]
    
    def postprocess(self, pred):
        """Verarbeite Vorhersagen und zeichne Ergebnisse"""
        # Tensor umformen
        pred = pred.squeeze(0)
        pred = pred.T
        
        # Confidence filtern
        pred = pred[pred[:, 4] > self.conf_threshold]
        
        # Boxen vorbereiten
        boxes = []
        scores = []
        classes = []
        
        for det in pred:
            x, y, w, h, conf, cls = det
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            width = int(w)
            height = int(h)
            
            boxes.append([x1, y1, width, height])
            scores.append(float(conf))
            classes.append(int(cls))
        
        # NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
        else:
            indices = []
        
        # Zeichnen
        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]
            x, y, w, h = boxes[i]
            conf = scores[i]
            cls = classes[i]
            
            cv2.rectangle(
                self.original_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                self.original_image,
                f"{cls}: {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return self.original_image
    
    def detect(self, image_path):
        """Kompletter Ablauf: Preprocess -> Infer -> Postprocess"""
        image = self.preprocess(image_path)
        pred = self.infer(image)
        result = self.postprocess(pred)
        return result


