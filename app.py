import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from gestures.gesture_detector import GestureDetector

# Page configuration
st.set_page_config(
    page_title="Gesture Detection AI",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .gesture-box {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">✋ Real-Time Gesture Detection AI</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence for hand detection"
    )
    
    tracking_confidence = st.slider(
        "Tracking Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for tracking"
    )
    
    show_landmarks = st.checkbox("Show Hand Landmarks", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    st.markdown("---")
    st.header("📚 Supported Gestures")
    st.markdown("""
    - 👍 **Thumbs Up**
    - 👎 **Thumbs Down**
    - ✌️ **Peace/Victory**
    - ✊ **Fist/Rock**
    - ✋ **Open Hand**
    - 👌 **OK Sign**
    - 🤟 **Rock On**
    - ☝️ **Pointing Up**
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Ensure good lighting and keep your hand clearly visible to the camera.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    FRAME_WINDOW = st.image([])
    
with col2:
    st.subheader("🎯 Detection Results")
    gesture_placeholder = st.empty()
    confidence_placeholder = st.empty()
    fps_placeholder = st.empty()
    stats_placeholder = st.empty()

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    run = st.button("▶️ Start Detection", type="primary", use_container_width=True)

with col_btn2:
    stop = st.button("⏹️ Stop Detection", use_container_width=True)

with col_btn3:
    snapshot = st.button("📸 Take Snapshot", use_container_width=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Button actions
if run:
    st.session_state.running = True
    st.session_state.detector = GestureDetector(
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=tracking_confidence
    )

if stop:
    st.session_state.running = False

# Main detection loop
if st.session_state.running and st.session_state.detector:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Unable to access webcam. Please check your camera permissions.")
        st.session_state.running = False
    else:
        # FPS calculation
        import time
        prev_time = time.time()
        
        while st.session_state.running:
            success, frame = cap.read()
            
            if not success:
                st.warning("⚠️ Unable to read from webcam.")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect gesture
            processed_frame, gesture, hand_confidence = st.session_state.detector.detect(
                frame, 
                draw_landmarks=show_landmarks
            )
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Add FPS to frame
            if show_fps:
                cv2.putText(
                    processed_frame, 
                    f"FPS: {int(fps)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
            
            # Display frame
            FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            # Display gesture info
            gesture_placeholder.markdown(
                f'<div class="gesture-box">{gesture}</div>', 
                unsafe_allow_html=True
            )
            
            if hand_confidence:
                confidence_placeholder.metric(
                    "Detection Confidence", 
                    f"{hand_confidence:.2%}"
                )
            
            if show_fps:
                fps_placeholder.metric("FPS", f"{int(fps)}")
            
            # Update stats
            st.session_state.frame_count += 1
            stats_placeholder.info(f"📊 Frames Processed: {st.session_state.frame_count}")
            
            # Take snapshot
            if snapshot:
                snapshot_path = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_path, processed_frame)
                st.success(f"✅ Snapshot saved as {snapshot_path}")
            
            # Check if stop was pressed
            if stop:
                break
        
        cap.release()
        cv2.destroyAllWindows()
else:
    # Show placeholder when not running
    placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        placeholder_image, 
        "Click 'Start Detection' to begin", 
        (100, 240), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 255), 
        2
    )
    FRAME_WINDOW.image(placeholder_image)
    gesture_placeholder.markdown(
        '<div class="gesture-box">No Gesture Detected</div>', 
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit, Mediapipe & OpenCV | 
        <a href='https://github.com' target='_blank'>GitHub</a></p>
    </div>
""", unsafe_allow_html=True)