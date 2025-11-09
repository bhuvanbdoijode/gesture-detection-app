"""
Utility functions for gesture detection and image processing.
"""

import numpy as np
import cv2
from typing import List, Tuple


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Normalize hand landmarks to a standard scale.
    
    Args:
        landmarks: Mediapipe hand landmarks
        
    Returns:
        Normalized coordinate array
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Center the coordinates
    coords -= np.mean(coords, axis=0)
    
    # Normalize by maximum distance from center
    max_norm = np.max(np.linalg.norm(coords, axis=1))
    if max_norm > 0:
        coords /= max_norm
    
    return coords


def calculate_distance(point1, point2) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y) or landmark
        point2: Second point (x, y) or landmark
        
    Returns:
        Euclidean distance
    """
    if hasattr(point1, 'x'):
        x1, y1 = point1.x, point1.y
    else:
        x1, y1 = point1
    
    if hasattr(point2, 'x'):
        x2, y2 = point2.x, point2.y
    else:
        x2, y2 = point2
    
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_angle(point1, point2, point3) -> float:
    """
    Calculate angle between three points (point2 is the vertex).
    
    Args:
        point1: First point
        point2: Vertex point
        point3: Third point
        
    Returns:
        Angle in degrees
    """
    # Extract coordinates
    if hasattr(point1, 'x'):
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
    else:
        p1, p2, p3 = np.array(point1), np.array(point2), np.array(point3)
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def preprocess_frame(frame: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
    """
    Preprocess video frame for better detection.
    
    Args:
        frame: Input frame
        size: Target size (width, height), optional
        
    Returns:
        Preprocessed frame
    """
    # Resize if size is specified
    if size is not None:
        frame = cv2.resize(frame, size)
    
    # Apply slight blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame


def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance frame contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        frame: Input BGR frame
        clip_limit: Contrast limiting threshold
        
    Returns:
        Contrast-enhanced frame
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def draw_bounding_box(frame: np.ndarray, landmarks, padding: int = 20) -> np.ndarray:
    """
    Draw bounding box around detected hand.
    
    Args:
        frame: Input frame
        landmarks: Hand landmarks
        padding: Padding around bounding box
        
    Returns:
        Frame with bounding box drawn
    """
    h, w, _ = frame.shape
    
    # Get all landmark coordinates
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]
    
    # Calculate bounding box
    x_min = max(0, min(x_coords) - padding)
    y_min = max(0, min(y_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_max = min(h, max(y_coords) + padding)
    
    # Draw rectangle
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return frame


def create_landmark_feature_vector(landmarks) -> np.ndarray:
    """
    Create a feature vector from hand landmarks for ML classification.
    
    Args:
        landmarks: Mediapipe hand landmarks
        
    Returns:
        Feature vector as numpy array
    """
    # Normalize landmarks first
    normalized = normalize_landmarks(landmarks)
    
    # Flatten to 1D vector (21 landmarks * 3 coordinates = 63 features)
    features = normalized.flatten()
    
    return features


def smooth_landmarks(current_landmarks: List, 
                     previous_landmarks: List, 
                     alpha: float = 0.5) -> List:
    """
    Apply exponential smoothing to landmarks for stable tracking.
    
    Args:
        current_landmarks: Current frame landmarks
        previous_landmarks: Previous frame landmarks
        alpha: Smoothing factor (0-1, higher = more responsive)
        
    Returns:
        Smoothed landmarks
    """
    if previous_landmarks is None:
        return current_landmarks
    
    smoothed = []
    for curr, prev in zip(current_landmarks, previous_landmarks):
        smoothed_x = alpha * curr.x + (1 - alpha) * prev.x
        smoothed_y = alpha * curr.y + (1 - alpha) * prev.y
        smoothed_z = alpha * curr.z + (1 - alpha) * prev.z
        
        # Create a new landmark-like object
        class SmoothedLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        smoothed.append(SmoothedLandmark(smoothed_x, smoothed_y, smoothed_z))
    
    return smoothed


def format_fps(fps: float) -> str:
    """
    Format FPS value for display.
    
    Args:
        fps: Frames per second
        
    Returns:
        Formatted string
    """
    if fps < 15:
        return f"⚠️ {fps:.1f} FPS (Low)"
    elif fps < 25:
        return f"✓ {fps:.1f} FPS (Good)"
    else:
        return f"✓ {fps:.1f} FPS (Excellent)"