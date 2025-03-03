import face_recognition
import cv2
import os


# Load known faces
def load_known_faces():
    known_encodings = []
    known_names = []

    for file in os.listdir("Dataset/"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = face_recognition.load_image_file(f"Dataset/{file}")
            encoding = face_recognition.face_encodings(img)[0]
            known_encodings.append(encoding)
            known_names.append(file.split(".")[0])

    return known_encodings, known_names


def recognize_face(image_path):
    known_encodings, known_names = load_known_faces()

    # Load test image
    img = face_recognition.load_image_file(image_path)
    test_encodings = face_recognition.face_encodings(img)

    for test_encoding in test_encodings:
        matches = face_recognition.compare_faces(known_encodings, test_encoding)

        if True in matches:
            match_index = matches.index(True)
            print(f"Recognized as: {known_names[match_index]}")
        else:
            print("Face not recognized.")


# Test function
recognize_face("Dataset/WhatsApp Image 2025-03-03 at 22.08.08.jpeg")
