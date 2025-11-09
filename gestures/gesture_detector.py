import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class GestureDetector:
    """
    Hand gesture detection using Mediapipe and OpenCV.
    Supports multiple gesture types with heuristic-based classification.
    """
    
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize the gesture detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        # Landmark indices
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.RING_TIP = 16
        self.RING_PIP = 14
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
        self.WRIST = 0
        
    def _is_finger_extended(self, landmarks, tip_idx, pip_idx) -> bool:
        """Check if a finger is extended."""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        return tip.y < pip.y
    
    def _is_thumb_extended(self, landmarks, handedness) -> bool:
        """Check if thumb is extended (special case due to thumb orientation)."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        
        # For right hand, thumb extends to the left; for left hand, to the right
        if handedness == "Right":
            return thumb_tip.x < thumb_ip.x
        else:
            return thumb_tip.x > thumb_ip.x
    
    def _count_extended_fingers(self, landmarks, handedness) -> int:
        """Count number of extended fingers."""
        count = 0
        
        # Check thumb (special case)
        if self._is_thumb_extended(landmarks, handedness):
            count += 1
        
        # Check other fingers
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if self._is_finger_extended(landmarks, tip, pip):
                count += 1
        
        return count
    
    def _classify_gesture(self, landmarks, handedness) -> str:
        """
        Classify hand gesture based on finger positions.
        
        Returns:
            String representation of detected gesture with emoji
        """
        # Get extended fingers
        thumb_extended = self._is_thumb_extended(landmarks, handedness)
        index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_extended = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        extended_count = self._count_extended_fingers(landmarks, handedness)
        
        # Thumbs Up - only thumb extended, pointing up
        if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            thumb_tip = landmarks[self.THUMB_TIP]
            index_mcp = landmarks[5]  # Index finger base
            if thumb_tip.y < index_mcp.y:
                return "👍 Thumbs Up"
            else:
                return "👎 Thumbs Down"
        
        # Peace/Victory - index and middle extended
        if index_extended and middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            return "✌️ Peace/Victory"
        
        # OK Sign - thumb and index forming circle
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        if distance < 0.05 and middle_extended and ring_extended and pinky_extended:
            return "👌 OK Sign"
        
        # Rock On - index and pinky extended
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "🤟 Rock On"
        
        # Pointing Up - only index extended
        if index_extended and not any([middle_extended, ring_extended, pinky_extended, thumb_extended]):
            return "☝️ Pointing Up"
        
        # Open Hand - all fingers extended
        if extended_count >= 4:
            return "✋ Open Hand"
        
        # Fist - no fingers extended
        if extended_count == 0:
            return "✊ Fist/Rock"
        
        # Default
        return f"🖐️ {extended_count} Fingers"
    
    def detect(self, frame, draw_landmarks=True) -> Tuple[np.ndarray, str, Optional[float]]:
        """
        Detect hand gestures in a frame.
        
        Args:
            frame: Input BGR image from camera
            draw_landmarks: Whether to draw hand landmarks on frame
            
        Returns:
            Tuple of (processed_frame, gesture_name, confidence)
        """
        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = "No Hand Detected"
        confidence = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # Process first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            confidence = results.multi_handedness[0].classification[0].score
            
            # Draw landmarks if requested
            if draw_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Classify gesture
            landmarks = hand_landmarks.landmark
            gesture = self._classify_gesture(landmarks, handedness)
            
            # Add gesture text to frame
            cv2.putText(
                frame,
                gesture,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )
        
        return frame, gesture, confidence
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'hands'):
            self.hands.close()