import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from Face_detection import detect_faces
from face_recognition_app import recognize_face, load_known_faces, start_webcam_recognition

# Load known faces once
known_encodings, known_names = load_known_faces()

# Create main application window
root = tk.Tk()
root.title("Face Recognition App")
root.geometry("600x500")


# Function to upload an image and detect faces
def upload_and_detect():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

    if not file_path:
        return

    # Display selected image
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk

    # Call the function from `face_detection.py`
    detect_faces(file_path)


# Function to recognize a face from an uploaded image
def upload_and_recognize():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

    if not file_path:
        return

    # Display selected image
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk

    # Call the function from `face_recognition_logic.py`
    result = recognize_face(file_path)
    messagebox.showinfo("Recognition Result", result)


# UI Elements
label_title = tk.Label(root, text="Face Recognition System", font=("Arial", 16))
label_title.pack(pady=10)

btn_detect = tk.Button(root, text="Detect Faces", command=upload_and_detect)
btn_detect.pack(pady=10)

btn_recognize = tk.Button(root, text="Recognize Face", command=upload_and_recognize)
btn_recognize.pack(pady=10)

btn_webcam = tk.Button(root, text="Start Live Recognition", command=start_webcam_recognition)
btn_webcam.pack(pady=10)

label_image = tk.Label(root)
label_image.pack(pady=10)

# Run the application
root.mainloop()
