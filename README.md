## YOLOPlants

In diesem Repository habe ich ein YOLO-Modell trainiert, um Pflanzen zu erkennen. Das Modell wurde mit einem Datensatz von Bildern trainiert, die verschiedene Pflanzenarten enthalten.

### Installation
Um das Modell zu verwenden, müssen Sie die folgenden Schritte ausführen:

1. Klonen Sie das Repository:
```bash
git clone https://github.com/Juljano/YOLOPlants.git
```
    
2. Installieren Sie die erforderlichen Abhängigkeiten:
```bash
pip install -r requirements.txt
```
### Verwendung
Starten Sie das Programm mit:
```bash 
python3 Main.py
``` 
Das Skript verwendet ein von mir trainiertes YOLO-Modell, um Pflanzen und Blumentöpfe in Bildern zu erkennen.

In der GUI können Sie eigene Bilder laden.
Das Modell analysiert diese Bilder und markiert erkannte Pflanzen und Blumentöpfe mit Bounding Boxes.

---
### Ergebnisse vom Training
Die folgende Tabelle zeigt die Ergebnisse des Trainingsprozesses:

| Class | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| ----- | ------ | --------- | ------------- | ---------- | ----- | -------- |
| all   | 40     | 84        | 0.822         | 0.535      | 0.613 | 0.268    |
| plant | 15     | 19        | 0.768         | 0.523      | 0.534 | 0.232    |
| pot   | 36     | 65        | 0.877         | 0.547      | 0.691 | 0.304    |
---

<img src="https://i.postimg.cc/J4j0HWwq/YOLOPlant.png" alt="Ein Foto von der Software" width="600" height=""/>

---

### Hinweis
- Die Erkennungsqualität hängt stark von den Trainingsdaten ab.
- Bei schwierigen Lichtverhältnissen oder ungewöhnlichen Perspektiven kann die Genauigkeit variieren.