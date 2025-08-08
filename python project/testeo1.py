
import cv2 
import mediapipe as mp

mp_drawning = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)