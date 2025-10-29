import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_y = {"index": None, "middle": None, "ring": None, "pinky": None}
THRESHOLD = 25

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                index_finger = handLms.landmark[8]
                middle_finger = handLms.landmark[12]
                ring_finger = handLms.landmark[16]
                pinky_finger = handLms.landmark[20]
                h, w, _ = frame.shape
                coords = {
                    "index": (int(index_finger.x * w), int(index_finger.y * h)),
                    "middle": (int(middle_finger.x * w), int(middle_finger.y * h)),
                    "ring": (int(ring_finger.x * w), int(ring_finger.y * h)),
                    "pinky": (int(pinky_finger.x * w), int(pinky_finger.y * h))
                }

                for name, (x, y) in coords.items():
                    if prev_y[name] is not None:
                        y = int(0.7 * prev_y[name] + 0.3 * y)
                        diff = prev_y[name] - y
                        if diff < -THRESHOLD:
                            cv2.putText(frame, f"{name} TAP!", (x-30, y-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    prev_y[name] = y

                    cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x-20, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("AirBeats - Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()