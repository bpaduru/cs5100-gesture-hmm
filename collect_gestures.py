# collect_gestures.py
# press 1-6 to record each gesture, q to quit

import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# gesture set - all six look completely different from each other
GESTURES = {
    0: "up",
    1: "right",
    2: "left",
    3: "down",
    4: "peace",
    5: "palm"
}

GESTURE_HINT = {
    0: "THUMBS UP - thumb pointing up, other fingers curled",
    1: "POINT RIGHT - index finger pointing horizontally to your right",
    2: "POINT LEFT - index finger pointing horizontally to your left",
    3: "THUMBS DOWN - thumb pointing down, other fingers curled",
    4: "PEACE SIGN - index and middle finger up in a V, others curled",
    5: "OPEN PALM - all five fingers spread flat facing camera"
}

KEY_MAP = {
    ord('1'): 0,
    ord('2'): 1,
    ord('3'): 2,
    ord('4'): 3,
    ord('5'): 4,
    ord('6'): 5
}

FRAMES_PER_SEQ = 20
TARGET_COUNT = 80
SAVE_DIR = "gesture_data_raw"
MODEL_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,5)
]

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found in current folder")
    print("download it by running:")
    print("curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit(1)

os.makedirs(SAVE_DIR, exist_ok=True)

saved_counts = {}
for gid, gname in GESTURES.items():
    count = len([f for f in os.listdir(SAVE_DIR) if f.startswith(gname + "_")])
    saved_counts[gid] = count

print("current saved counts:")
for gid, gname in GESTURES.items():
    status = "DONE" if saved_counts[gid] >= TARGET_COUNT else f"{saved_counts[gid]}/{TARGET_COUNT}"
    print(f"  {gid+1} = {gname}: {status}")

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

recording = False
active_gesture = None
buffer = []
warmup_frames = 0

print("\nkeys: 1=up  2=right  3=left  4=down  5=peace  6=palm  q=quit")
print("press a key then hold the gesture until it auto-saves\n")


def draw_hand(frame, lm_list, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 180, 0), 1)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    hand_landmarks = None
    if result.hand_landmarks:
        lm_list = result.hand_landmarks[0]
        draw_hand(frame, lm_list, H, W)
        hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])

    for gid, gname in GESTURES.items():
        done = saved_counts[gid] >= TARGET_COUNT
        color = (0, 180, 0) if done else (0, 0, 200)
        cv2.putText(frame, f"{gid+1}={gname} {saved_counts[gid]}/{TARGET_COUNT}",
                    (10, 28 + gid * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    if recording:
        if warmup_frames > 0:
            cv2.putText(frame, f"get ready: {GESTURES[active_gesture]}",
                        (10, H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, GESTURE_HINT[active_gesture],
                        (10, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
            warmup_frames -= 1
        else:
            if hand_landmarks is not None:
                buffer.append(hand_landmarks.copy())
                cv2.putText(frame,
                            f"recording {GESTURES[active_gesture]}  {len(buffer)}/{FRAMES_PER_SEQ}",
                            (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "show hand in frame!", (10, H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if len(buffer) >= FRAMES_PER_SEQ:
                gname = GESTURES[active_gesture]
                idx = saved_counts[active_gesture]
                path = os.path.join(SAVE_DIR, f"{gname}_{idx:03d}.npy")
                np.save(path, np.array(buffer))
                saved_counts[active_gesture] += 1
                print(f"saved {path}  ({saved_counts[active_gesture]}/{TARGET_COUNT})")
                recording = False
                active_gesture = None
                buffer = []

    cv2.imshow("gesture collection - click this window then press 1-6", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in KEY_MAP and not recording:
        gid = KEY_MAP[key]
        if saved_counts[gid] < TARGET_COUNT:
            active_gesture = gid
            recording = True
            buffer = []
            warmup_frames = 25
            print(f"starting {GESTURES[gid]}: {GESTURE_HINT[gid]}")
        else:
            print(f"{GESTURES[gid]} already done")

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("\ncollection done. check gesture_data_raw/ folder")
print("next step: run features.py")
