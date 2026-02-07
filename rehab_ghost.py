import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import pyttsx3

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Colors (BGR)
COLORS = {
    "Cyan":    (255, 255, 0),
    "Green":   (0, 255, 0),
    "Yellow":  (0, 255, 255),
    "Red":     (0, 0, 255),
    "Blue":    (255, 0, 0),
    "Purple":  (255, 0, 255),
    "Orange":  (0, 165, 255),
    "Pink":    (203, 192, 255),
    "White":   (255, 255, 255),
    "Lime":    (150, 255, 0)
}

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# ─────────────────────────────────────────────────────────────────────────────
# Ideal normalized landmark positions for ghost trainer (0–1 range)
# These are rough but representative "perfect" poses

IDEAL_POSES = {
    "Squat": {
        # Bottom of squat (deep position, thighs roughly parallel)
        mp_pose.PoseLandmark.LEFT_HIP.value:     (0.48, 0.58, 0),
        mp_pose.PoseLandmark.LEFT_KNEE.value:    (0.48, 0.82, 0),
        mp_pose.PoseLandmark.LEFT_ANKLE.value:   (0.48, 0.98, 0),
        mp_pose.PoseLandmark.RIGHT_HIP.value:    (0.52, 0.58, 0),
        mp_pose.PoseLandmark.RIGHT_KNEE.value:   (0.52, 0.82, 0),
        mp_pose.PoseLandmark.RIGHT_ANKLE.value:  (0.52, 0.98, 0),
        mp_pose.PoseLandmark.LEFT_SHOULDER.value: (0.48, 0.32, 0),
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value:(0.52, 0.32, 0),
    },
    "Bicep Curl": {
        # Top of curl (elbow bent, weight near shoulder)
        mp_pose.PoseLandmark.LEFT_SHOULDER.value:  (0.45, 0.28, 0),
        mp_pose.PoseLandmark.LEFT_ELBOW.value:     (0.42, 0.55, 0),
        mp_pose.PoseLandmark.LEFT_WRIST.value:     (0.40, 0.72, 0),
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value: (0.55, 0.28, 0),
        mp_pose.PoseLandmark.RIGHT_ELBOW.value:    (0.58, 0.55, 0),
        mp_pose.PoseLandmark.RIGHT_WRIST.value:    (0.60, 0.72, 0),
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# TTS
@st.cache_resource
def get_tts_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        voices = engine.getProperty('voices')
        for v in voices:
            if "english" in v.name.lower():
                engine.setProperty('voice', v.id)
                break
        return engine
    except:
        return None

def speak(engine, text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

# ─────────────────────────────────────────────────────────────────────────────
st.title("Rehab-Visor • AI Physical Therapy Coach")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")

model_complexity = st.sidebar.selectbox("Model", ["Light", "Optimal", "Heavy"], index=1)
model_complexity = ["Light", "Optimal", "Heavy"].index(model_complexity)

min_det = st.sidebar.slider("Detection conf", 0.1, 0.9, 0.50, 0.05)
min_trk = st.sidebar.slider("Tracking conf", 0.1, 0.9, 0.50, 0.05)

joint_color = COLORS[st.sidebar.selectbox("Joint color", list(COLORS.keys()), index=0)]
bone_color  = COLORS[st.sidebar.selectbox("Bone color",  list(COLORS.keys()), index=0)]

exercise = st.sidebar.selectbox("Exercise", ["None", "Squat", "Bicep Curl"])

video_source = st.sidebar.radio("Source", ["Webcam", "Upload Video"])
uploaded_video = None
if video_source == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Choose video", type=["mp4","avi","mov"])

enable_tts = st.sidebar.checkbox("Voice feedback", value=True)
ghost_enabled = st.sidebar.checkbox("Show Ghost Trainer", value=True)
ghost_alpha = st.sidebar.slider("Ghost opacity", 0.1, 1.0, 0.5, 0.05)

save_video = st.sidebar.checkbox("Record output video")
output_path = st.sidebar.text_input("Output path", "rehab_output.avi") if save_video else None

# ── Session state ──────────────────────────────────────────────────────────
if 'running' not in st.session_state:
    st.session_state.running = False
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = get_tts_engine()

frame_placeholder = st.image([])
feedback_placeholder = st.empty()
fps_placeholder = st.sidebar.empty()

col1, col2 = st.columns(2)
if col1.button("Start"):
    st.session_state.running = True
if col2.button("Stop"):
    st.session_state.running = False

# ── Main processing loop ───────────────────────────────────────────────────
if st.session_state.running:
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        with open("temp_upload.mp4", "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture("temp_upload.mp4")
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open video source")
        st.session_state.running = False
    else:
        out_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            w, h = int(cap.get(3)), int(cap.get(4))
            out_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

        prev_time = time.time()
        frame_count = 0

        with mp_pose.Pose(
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk,
            model_complexity=model_complexity
        ) as pose:

            while st.session_state.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # mirror
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                feedback = ""
                fb_color = (255, 255, 255)

                if results.pose_landmarks:
                    # Draw user skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=joint_color, thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=bone_color, thickness=2)
                    )

                    lm = results.pose_landmarks.landmark

                    # ── Exercise logic ───────────────────────────────────────
                    if exercise != "None":
                        try:
                            if exercise == "Squat":
                                hip   = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                knee  = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,  lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                                ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                                angle = calculate_angle(hip, knee, ankle)

                                if angle > 160:
                                    st.session_state.stage = "up"
                                if angle <= 90 and st.session_state.stage == "up":
                                    st.session_state.stage = "down"
                                    st.session_state.rep_count += 1
                                    feedback = f"Good squat!  Rep {st.session_state.rep_count}"
                                    fb_color = (0, 255, 0)
                                    if enable_tts:
                                        speak(st.session_state.tts_engine, "Great squat!")

                            elif exercise == "Bicep Curl":
                                shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                elbow    = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,    lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                                wrist    = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,    lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                                angle    = calculate_angle(shoulder, elbow, wrist)

                                if angle > 160:
                                    st.session_state.stage = "down"
                                if angle <= 35 and st.session_state.stage == "down":
                                    st.session_state.stage = "up"
                                    st.session_state.rep_count += 1
                                    feedback = f"Good curl!  Rep {st.session_state.rep_count}"
                                    fb_color = (0, 255, 0)
                                    if enable_tts:
                                        speak(st.session_state.tts_engine, "Excellent curl!")

                        except:
                            pass

                    # ── Draw Ghost Trainer ───────────────────────────────────
                    if ghost_enabled and exercise in IDEAL_POSES:
                        ghost_overlay = frame.copy()
                        ideal = IDEAL_POSES[exercise]

                        ghost_line_color = (0, 220, 255)   # bright yellow
                        ghost_dot_color  = (50, 180, 255)

                        for conn in mp_pose.POSE_CONNECTIONS:
                            a_idx, b_idx = conn
                            if a_idx in ideal and b_idx in ideal:
                                pa = ideal[a_idx]
                                pb = ideal[b_idx]
                                start = (int(pa[0]*frame.shape[1]), int(pa[1]*frame.shape[0]))
                                end   = (int(pb[0]*frame.shape[1]), int(pb[1]*frame.shape[0]))
                                cv2.line(ghost_overlay, start, end, ghost_line_color, 5)

                        for idx, pos in ideal.items():
                            pt = (int(pos[0]*frame.shape[1]), int(pos[1]*frame.shape[0]))
                            cv2.circle(ghost_overlay, pt, 7, ghost_dot_color, -1)

                        # Blend
                        cv2.addWeighted(ghost_overlay, ghost_alpha, frame, 1 - ghost_alpha, 0, frame)

                # ── UI overlays ──────────────────────────────────────────────
                cv2.putText(frame, f"Reps: {st.session_state.rep_count}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, fb_color, 3)
                if feedback:
                    cv2.putText(frame, feedback, (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, fb_color, 2)

                # FPS
                frame_count += 1
                now = time.time()
                if now - prev_time >= 1:
                    fps_placeholder.text(f"FPS: {frame_count}")
                    frame_count = 0
                    prev_time = now

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if out_writer:
                    out_writer.write(frame)

                time.sleep(0.015)

        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        if uploaded_video:
            try:
                os.remove("temp_upload.mp4")
            except:
                pass

        st.session_state.running = False