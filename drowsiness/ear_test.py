import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# MediaPipe FaceMesh landmark indices for eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = coords
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

EAR_THRESHOLD = 0.21  # below this = eyes likely closed, tune after testing

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        status = "DROWSY/CLOSED" if avg_ear < EAR_THRESHOLD else "ALERT"
        color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, status, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("EAR Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()