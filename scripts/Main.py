import tkinter
from tkinter import Tk, filedialog
from PIL import Image, ImageTk
from YOLODetector import YOLODetector


# @Juljano Mario Möller
class Main:
    def __init__(self):
        self.tk = Tk()
        self.tk.title("Plant and Pots Detecting")
        self.tk.geometry("600x600")
        self.open_Button = tkinter.Button(self.tk, text="Select a Image file", command=self.open_file_dialog)
        self.open_Button.pack(padx=20, pady=20, anchor="center")
        self.panel = tkinter.Label(self.tk, bg="gray")
        self.panel.pack(padx=20, pady=20, fill="both", expand=True)
        self.imageview = None
        self.tk.mainloop()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select a image file",filetypes=[
            ("JPG Files",".jpg"),
            ("JPEG Files", ".jpeg"),
            ("PNG Files","*.png"),
            ("WebP Files","*.webp")
        ])
        if file_path:
            self.loading_yolo_model(file_path)
            return file_path
        return None

    def display_image(self, photo):
        try:
            pil_image = Image.fromarray(photo)
            max_size = (500, 400)
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            self.imageview = ImageTk.PhotoImage(pil_image)
            self.panel.image = self.imageview
            self.panel.pack(padx=20, pady=20, anchor="center")
            self.panel.configure(image=self.imageview)
        except Exception as error:
            print(f"Error displaying image: {error}")


    def loading_yolo_model(self,path):
        if path is None:
            print("Path is required")
            return
        try:
            result = detector.detect(path)
            self.display_image(result)
        except Exception as error:
            print(f"Error loading YOLO model: {error}")



if __name__ == "__main__":
    detector = YOLODetector()
    main = Main()


