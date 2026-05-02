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
Es wird zusätzlich auch im GUI die erkannten Klassen angezeigt

<img alt="Ein Foto von der Software" src="https://i.postimg.cc/J4j0HWwq/YOLOPlant.png" width="500">

---

### Hinweis
- Das Modell basiert auf dem YOLO-Framework.
- Die Erkennungsqualität hängt stark von den Trainingsdaten ab.
- Bei schwierigen Lichtverhältnissen oder ungewöhnlichen Perspektiven kann die Genauigkeit variieren.
