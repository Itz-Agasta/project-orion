import time

STATE_IDLE = 0
STATE_TRACKING = 1

class Tracker:
    """
    Contains the state management logic for:
      - Idle: waiting for activation gesture.
      - Tracking: locked onto the arm corresponding to the hand used for activation.
    """
    def __init__(self):
        self.state = STATE_IDLE
        self.activation_start_time = None  
        self.lost_start_time = None
        self.tracked_hand = None   # Hand landmarks used for activation.
        self.tracked_box_area = None
        self.tracked_hand_side = None  # "Left" or "Right", after mirroring.

    def reset(self):
        self.state = STATE_IDLE
        self.activation_start_time = None
        self.lost_start_time = None
        self.tracked_hand = None
        self.tracked_box_area = None
        self.tracked_hand_side = None
