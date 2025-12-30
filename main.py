import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Image/Video Face Recognition")
        master.geometry('500x400')
        
        self.upload_button = tk.Button(master, text="Upload Image/Video", command=self.upload_file, font=("Arial", 14))
        self.upload_button.pack(pady=20)
        
        self.quit_label = tk.Label(master, text="Press 'Q' to quit", font=("Arial", 12))
        self.quit_label.pack(pady=5)

        self.result_label = tk.Label(master)
        self.result_label.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png"), ("Video files", "*.mp4;*.avi")])
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if file_path.endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.detect_faces(frame, face_cascade)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        elif file_path.endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(file_path)
            if image is not None:
                self.detect_faces(image, face_cascade)
                cv2.imshow('Image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                messagebox.showerror("Error", "Unable to load image.")

    def detect_faces(self, frame, face_cascade):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()