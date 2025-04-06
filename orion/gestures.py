def detect_activation_gesture(hand_landmarks):
    """
    Returns True if the hand shows:
      - Index (8) and middle (12) fingers extended (y-coordinate lower than their preceding joints),
      - Ring (16) and pinky (20) fingers curled.
    """
    landmarks = hand_landmarks.landmark
    index_extended = landmarks[8].y < landmarks[7].y
    middle_extended = landmarks[12].y < landmarks[11].y
    ring_curled = landmarks[16].y > landmarks[15].y
    pinky_curled = landmarks[20].y > landmarks[19].y
    return index_extended and middle_extended and ring_curled and pinky_curled
