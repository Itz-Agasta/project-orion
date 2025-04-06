import cv2
import mediapipe as mp
from orion.tracker import Tracker
from orion.vision import process_frame

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.8,
                                        min_tracking_confidence=0.5)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.8,
                                     min_tracking_confidence=0.5)
    tracker = Tracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, mp_hands, mp_pose, tracker)
        cv2.imshow('Orion Vision', processed_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
