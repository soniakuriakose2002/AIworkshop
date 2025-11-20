import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import asyncio

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.7)

# Constants
KNOWN_WIDTH = 8.0
FOCAL_LENGTH = 500

engine = pyttsx3.init()

def calculate_distance(perceived_width):
    if perceived_width == 0:
        return 0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

async def speak_async(text):
    await asyncio.to_thread(engine.say, text)
    await asyncio.to_thread(engine.runAndWait)

async def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = 0, 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                perceived_width = x_max - x_min
                distance = calculate_distance(perceived_width)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'Distance: {distance:.2f} cm', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if distance <= 20:
                    asyncio.create_task(speak_async("Too close"))

        cv2.imshow('Hand Distance Measurement', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Hand Distance Measurement', cv2.WND_PROP_VISIBLE) < 1:
            break

        await asyncio.sleep(0)

    cap.rel