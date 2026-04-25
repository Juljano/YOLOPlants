import tkinter
from tkinter import Tk, filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk


class Main:
    def __init__(self):
        self.tk = Tk()
        self.tk.title("Detecting with YOLO")
        self.tk.geometry("600x600")
        self.open_Button = tkinter.Button(self.tk, text="Select a Image file", command=self.open_file_dialog)
        self.open_Button.pack(padx=20, pady=20, anchor="center")
        self.panel = tkinter.Label(self.tk, bg="gray")
        self.panel.pack(padx=20, pady=20, fill="both", expand=True)
        self.imageview = None
        self.tk.mainloop()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select a image file",filetypes=[("Image file",".jpg")])
        if file_path is not None:
            print(f"Selected file: {file_path}")
            self.loading_yolo_model(file_path)
            return file_path
        return None

    def display_image(self, photo):
        try:
            pil_image = Image.fromarray(photo)
            max_size = (500, 500)
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
            model = YOLO("model/best.pt")
            result = model(path, conf=0.5)
            model_pred = result[0].plot()
            self.display_image(model_pred)
        except Exception as error:
            print(f"Error loading YOLO model: {error}")



if __name__ == "__main__":
    main = Main()

