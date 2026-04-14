import cv2
import face_recognition
import os
import numpy as np

# Load known faces
known_face_encodings = []
known_face_names = []

folder = "known_faces"

for filename in os.listdir(folder):

    if filename.endswith(".jpg"):

        image_path = os.path.join(folder, filename)

        image = face_recognition.load_image_file(
            image_path
        )

        encoding = face_recognition.face_encodings(
            image
        )[0]

        known_face_encodings.append(
            encoding
        )

        name = os.path.splitext(filename)[0]

        known_face_names.append(
            name
        )

print("Loaded faces:", known_face_names)

# Load eye detector
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# Start webcam
cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Cannot open webcam")
    exit()

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        name = "Unknown"

        distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(distances) > 0:

            best_match_index = np.argmin(
                distances
            )

            if matches[best_match_index]:

                name = known_face_names[
                    best_match_index
                ]

        # Draw face box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        # Eye tracking
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        roi_gray = gray[
            top:bottom,
            left:right
        ]

        roi_color = frame[
            top:bottom,
            left:right
        ]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

            center_x = ex + ew // 2
            center_y = ey + eh // 2

            cv2.circle(
                roi_color,
                (center_x, center_y),
                3,
                (0, 0, 255),
                -1
            )

            print(
                "Eye position:",
                center_x,
                center_y
            )

    cv2.imshow(
        "Face Recognition + Eye Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()