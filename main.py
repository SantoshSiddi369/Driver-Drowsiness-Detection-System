# main.py
import cv2
import numpy as np
import os
import sys
from utils.sound_alert import play_alert

# Paths to Haar cascade models and alert sound
FACE_CASCADE_PATH = os.path.join('models', 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join('models', 'haarcascade_eye.xml')
ALERT_SOUND_PATH = os.path.join('models', 'alert.wav')

# Check if Haar cascade files exist
if not os.path.exists(FACE_CASCADE_PATH):
    print(f"Error: {FACE_CASCADE_PATH} not found.")
    sys.exit(1)
if not os.path.exists(EYE_CASCADE_PATH):
    print(f"Error: {EYE_CASCADE_PATH} not found.")
    sys.exit(1)


face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
if face_cascade.empty():
    print(f'Error loading {FACE_CASCADE_PATH}. File may be corrupted or not a valid Haar cascade.')
    sys.exit(1)
if eye_cascade.empty():
    print(f'Error loading {EYE_CASCADE_PATH}. File may be corrupted or not a valid Haar cascade.')
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Cannot access webcam.')
    sys.exit(1)


score = 0
SCORE_THRESHOLD = 900  # 30 seconds at ~30 FPS
alert_played = False
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame.')
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_detected = False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Requirement 1: Face not detected
        if len(faces) == 0:
            score = 0
            alert_played = False
            cv2.putText(frame, 'Face is not detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            # Requirement 2: Eyes open
            if eyes_detected:
                score = 0
                alert_played = False
                cv2.putText(frame, 'Eyes are open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                score += 1
                remaining = max(0, SCORE_THRESHOLD - score)
                cv2.putText(frame, f'Eyes closed: {score//30}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # Requirement 3: Drowsiness detected after 30 seconds
                if score >= SCORE_THRESHOLD:
                    if not alert_played:
                        try:
                            play_alert(ALERT_SOUND_PATH)
                        except Exception as e:
                            print(f"[Warning] Could not play alert sound: {e}")
                            # Fallback: system beep
                            if sys.platform == "win32":
                                import winsound
                                winsound.Beep(1000, 500)
                            else:
                                print("\a")  # ASCII bell
                        alert_played = True
                    cv2.putText(frame, 'Drowsiness detected!', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

