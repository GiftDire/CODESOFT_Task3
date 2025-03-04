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


# Recognize a face from an uploaded image
def recognize_face(image_path):
    known_encodings, known_names = load_known_faces()
    img = face_recognition.load_image_file(image_path)
    test_encodings = face_recognition.face_encodings(img)

    for test_encoding in test_encodings:
        matches = face_recognition.compare_faces(known_encodings, test_encoding)

        if True in matches:
            match_index = matches.index(True)
            return f"Recognized as: {known_names[match_index]}"
        else:
            return "Face not recognized."


# Live face recognition from your webcam
def start_webcam_recognition():
    known_encodings, known_names = load_known_faces()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show webcam feed
        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
