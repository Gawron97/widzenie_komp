import cv2
import pandas as pd
import numpy as np
import os

VIDEO_PATH = "anon/anon/044_b5f6wze1_.mp4"
CSV_PATH = os.path.basename(VIDEO_PATH).replace('.mp4', '.csv')
OUTPUT_VIDEO_PATH = "pose_visualization.mp4"

COCO_CONNECTIONS_NAMES = [
    ("NOSE", "LEFT_EYE"), ("NOSE", "RIGHT_EYE"), ("LEFT_EYE", "LEFT_EAR"), ("RIGHT_EYE", "RIGHT_EAR"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE")
]

try:
    df = pd.read_csv(CSV_PATH)
    grouped = df.groupby('frame_number')
    frame_data_map = {
        frame_num: frame_group.set_index('landmark')
        for frame_num, frame_group in grouped
    }
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    print(f'Error')
    cap.release()
    exit()

frame_count = 0
is_paused = False
current_frame_num = 0

while cap.isOpened():
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_num = frame_count
        frame_count += 1
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
        ret, frame = cap.read()
        if not ret:
            break

    frame_landmarks = frame_data_map.get(current_frame_num)

    if frame_landmarks is not None:
        points_px = {}

        try:
            sample_x = frame_landmarks[frame_landmarks['x'] > 0]['x'].iloc[0]
            is_normalized = sample_x <= 1.0
        except IndexError:
            is_normalized = True

        for name, row in frame_landmarks.iterrows():
            x, y = row['x'], row['y']
            if x == 0 and y == 0:
                points_px[name] = (0, 0)
            else:
                if is_normalized:
                    px = int(x * width)
                    py = int(y * height)
                else:
                    px = int(x)
                    py = int(y)
                points_px[name] = (px, py)
        
        for p1_name, p2_name in COCO_CONNECTIONS_NAMES:
            p1 = points_px.get(p1_name)
            p2 = points_px.get(p2_name)
            if p1 and p2 and p1 != (0, 0) and p2 != (0, 0):
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

        for name, (px, py) in points_px.items():
            if (px, py) != (0, 0):
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    cv2.putText(frame, f'Frame: {current_frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # cv2.imshow('Pose Visualization', frame)
    out.write(frame)

    wait_time = int(1000 / fps)
    key = cv2.waitKey(wait_time) & 0xFF

    if key == ord('q'):
        break
    if key == ord('p'):
        is_paused = not is_paused
    
out.release()
cap.release()
cv2.destroyAllWindows()