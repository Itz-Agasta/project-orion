def compute_bounding_box(hand_landmarks, width, height):
    xs = [lm.x * width for lm in hand_landmarks.landmark]
    ys = [lm.y * height for lm in hand_landmarks.landmark]
    return min(xs), min(ys), max(xs), max(ys)

def compute_box_area(box):
    min_x, min_y, max_x, max_y = box
    return (max_x - min_x) * (max_y - min_y)

#FIXME: this is a bit of a hack, though it works for now but find a better way to do this without mirroring.
def compute_arm_vector(pose_landmarks, width, height, side):
    """
    Computes the arm vector based on pose landmarks for a mirrored (selfie) image.
    
    In selfie mode, the image is flipped horizontally, so the landmarks for the right 
    and left sides are swapped. This function accounts for the mirroring to correctly 
    compute the arm vector.

    original landmarks for right side:
      - Right shoulder: landmark 12
      - Right elbow: landmark 14
      - Right wrist: landmark 16

    Original landmarks for left side:
      - Left shoulder: landmark 11
      - Left elbow: landmark 13
      - Left wrist: landmark 15
    """
    if side.lower() == 'left':
        shoulder_lm = pose_landmarks.landmark[12]
        elbow_lm = pose_landmarks.landmark[14]
        wrist_lm = pose_landmarks.landmark[16]
    else:
        shoulder_lm = pose_landmarks.landmark[11]
        elbow_lm = pose_landmarks.landmark[13]
        wrist_lm = pose_landmarks.landmark[15]
    
    shoulder = (int(shoulder_lm.x * width), int(shoulder_lm.y * height))
    elbow = (int(elbow_lm.x * width), int(elbow_lm.y * height))
    wrist = (int(wrist_lm.x * width), int(wrist_lm.y * height))
    vector = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    return shoulder, elbow, wrist, vector

def draw_arm(image, shoulder, elbow, wrist):
    import cv2
    cv2.circle(image, shoulder, 5, (0, 255, 0), -1)
    cv2.circle(image, elbow, 5, (0, 255, 0), -1)
    cv2.circle(image, wrist, 5, (0, 255, 0), -1)
    cv2.line(image, shoulder, elbow, (255, 0, 0), 3)
    cv2.line(image, elbow, wrist, (255, 0, 0), 3)
