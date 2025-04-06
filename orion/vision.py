import cv2
import time
import mediapipe as mp
from .gestures import detect_activation_gesture
from .utils import compute_bounding_box, compute_box_area, compute_arm_vector, draw_arm
from .tracker import STATE_IDLE, STATE_TRACKING

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def process_frame(image, hands, pose, tracker):
    """
    Process a video frame for hand gesture detection and arm tracking.
    
    Args:
        image: BGR image from camera
        hands: MediaPipe hands solution instance
        pose: MediaPipe pose solution instance
        tracker: Tracker object maintaining the current state
    
    Returns:
        Processed image with visualizations based on current tracking state
    """
    # Flip image for a selfie-view and convert to RGB.
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Run hand and pose detections.
    image.flags.writeable = False
    hand_results = hands.process(rgb_image)
    pose_results = pose.process(rgb_image)
    image.flags.writeable = True

    activation_candidates = []         # List to store hand candidates for activation gesture
    
    # In tracking mode, use the tracked hand side
    side_to_track = tracker.tracked_hand_side.lower() if tracker.state == STATE_TRACKING and tracker.tracked_hand_side else 'none'
    
    # In tracking mode: draw arm landmarks and the aim vector.
    if tracker.state == STATE_TRACKING and pose_results.pose_landmarks:
        shoulder, elbow, wrist, arm_vector = compute_arm_vector(pose_results.pose_landmarks, width, height, side_to_track)
        draw_arm(image, shoulder, elbow, wrist)
        endpoint = (wrist[0] + arm_vector[0], wrist[1] + arm_vector[1])
        cv2.arrowedLine(image, wrist, endpoint, (0, 255, 255), 3)
        cv2.putText(image, f"Arm Vector: {arm_vector}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"Pointing Coordinates: {endpoint}")
    
    # In idle mode: draw hand landmarks.
    if tracker.state == STATE_IDLE and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
    
    # Process hand landmarks for gesture detection (without drawing in tracking mode).
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Get handedness from MediaPipe.
            handedness = hand_results.multi_handedness[idx].classification[0].label  # "Left" or "Right"
            if detect_activation_gesture(hand_landmarks):
                box = compute_bounding_box(hand_landmarks, width, height)
                area = compute_box_area(box)
                activation_candidates.append((hand_landmarks, area, box, handedness))
                if tracker.state == STATE_IDLE:
                    cv2.rectangle(image,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 255, 0), 2)
    
    # Display current mode on screen.         
    mode_text = f"Mode: Tracking ({tracker.tracked_hand_side})" if tracker.state == STATE_TRACKING else "Mode: Idle"
    cv2.putText(image, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # ----- State Machine Handling -----
    if tracker.state == STATE_IDLE:
        if activation_candidates:
            candidate = max(activation_candidates, key=lambda x: x[1])
            if tracker.activation_start_time is None:
                tracker.activation_start_time = time.time()
                tracker.tracked_hand = candidate[0]
                tracker.tracked_box_area = candidate[1]
                tracker.tracked_hand_side = candidate[3]
            else:
                # Check if the candidate remains similar.
                if abs(candidate[1] - tracker.tracked_box_area) / tracker.tracked_box_area < 0.2:
                    elapsed = time.time() - tracker.activation_start_time
                    cv2.putText(image, f"Activating: {elapsed:.1f}s", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if elapsed >= 1.5:
                        tracker.state = STATE_TRACKING
                        tracker.activation_start_time = None
                        cv2.putText(image, "Tracking Activated", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    tracker.activation_start_time = time.time()
                    tracker.tracked_hand = candidate[0]
                    tracker.tracked_box_area = candidate[1]
                    tracker.tracked_hand_side = candidate[3]
        else:
            tracker.activation_start_time = None
            tracker.tracked_hand = None
            tracker.tracked_box_area = None
            tracker.tracked_hand_side = None
            cv2.putText(image, "Waiting for Activation Gesture", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    elif tracker.state == STATE_TRACKING:
        # Check for hand gesture to turn off tracking.
        if activation_candidates:
            candidate = max(activation_candidates, key=lambda x: x[1])
            if tracker.activation_start_time is None:
                tracker.activation_start_time = time.time()
            else:
                elapsed = time.time() - tracker.activation_start_time
                cv2.putText(image, f"Deactivating: {elapsed:.1f}s", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if elapsed >= 1.5:
                    tracker.reset()
                    cv2.putText(image, "Tracking Deactivated", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            tracker.activation_start_time = None

        # If pose landmarks are lost, start a lost timer.
        if not pose_results.pose_landmarks:
            if tracker.lost_start_time is None:
                tracker.lost_start_time = time.time()
            elif time.time() - tracker.lost_start_time >= 3:
                tracker.reset()
                cv2.putText(image, "Tracking Lost", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            tracker.lost_start_time = None

    return image

