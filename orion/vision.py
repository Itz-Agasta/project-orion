import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def process_gesture(hand_landmarks, image, width, height):
    """
    Processes hand landmarks to check for the specific gesture:
    - Index (8) and middle (12) fingers extended.
    - Ring (16) and pinky (20) fingers curled.
    
    If the gesture is detected, calculates and returns the midpoint between
    the index and middle fingertips (in pixel coordinates).
    """
    landmarks = hand_landmarks.landmark

    # Check finger states using y-coordinates.
    index_extended = landmarks[8].y < landmarks[7].y
    middle_extended = landmarks[12].y < landmarks[11].y
    ring_curled = landmarks[16].y > landmarks[15].y
    pinky_curled = landmarks[20].y > landmarks[19].y

    # If the gesture matches: index and middle extended, ring and pinky curled.
    if index_extended and middle_extended and ring_curled and pinky_curled:
        index_tip_pixel = (int(landmarks[8].x * width), int(landmarks[8].y * height))
        middle_tip_pixel = (int(landmarks[12].x * width), int(landmarks[12].y * height))

        x_mid = (index_tip_pixel[0] + middle_tip_pixel[0]) // 2
        y_mid = (index_tip_pixel[1] + middle_tip_pixel[1]) // 2
        return (x_mid, y_mid)
    
    return None

def draw_hand_annotations(image, hand_landmarks):
    """
    Draws landmarks and connections on the image using MediaPipe utilities.
    """
    mp_drawing.draw_landmarks(
        image, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
    )

def process_frame(image, hands):
    """
    Processes a single frame:
    - Converts colors and flips the image.
    - Detects hand landmarks.
    - Processes each hand for gesture detection and draws annotations.
    
    Returns the processed image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_hand_annotations(image, hand_landmarks)
            
            midpoint = process_gesture(hand_landmarks, image, width, height)
            if midpoint:
                cv2.circle(image, midpoint, 10, (0, 0, 255), -1)
                cv2.putText(image, "Gesture Detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                print(f"Aim Turret at: {midpoint}") # Debugging output
    
    return image

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, hands)
            cv2.imshow('Hand Tracking', processed_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
