# mediapipe_baseline.py
# rule-based classifier using hand landmark geometry
# works on the same raw gesture_data_raw files


import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]
RAW_DIR = "gesture_data_raw"


def classify_one_frame(landmarks):
    # landmarks is (21, 3), returns 0-5 or -1 if unclear
    wrist     = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]

    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    # finger is extended if its tip is above pip in the image (lower y value)
    finger_ext = [landmarks[tip_ids[i]][1] < landmarks[pip_ids[i]][1]
                  for i in range(1, 5)]

    thumb_ext = abs(thumb_tip[0] - wrist[0]) > abs(thumb_mcp[0] - wrist[0])
    n_ext = sum(finger_ext)

    # all fingers open = palm
    if n_ext >= 3:
        return 5

    # exactly index and middle up = peace sign
    if finger_ext[0] and finger_ext[1] and not finger_ext[2] and not finger_ext[3]:
        return 4

    # thumb extended, all fingers curled = thumbs up or down
    if thumb_ext and n_ext == 0:
        if thumb_tip[1] < wrist[1]:
            return 0  # thumbs up = up
        else:
            return 3  # thumbs down = down

    # only index extended = pointing
    if finger_ext[0] and not any(finger_ext[1:]):
        dx = index_tip[0] - index_mcp[0]
        dy = index_tip[1] - index_mcp[1]
        if abs(dx) > abs(dy):
            if dx > 0:
                return 1  # pointing right
            else:
                return 2  # pointing left

    return -1


def classify_sequence(raw_seq):
    frame_preds = [classify_one_frame(frame) for frame in raw_seq]
    valid = [p for p in frame_preds if p != -1]
    if not valid:
        return -1
    return Counter(valid).most_common(1)[0][0]


all_preds = []
all_true = []

for g_idx, gname in enumerate(GESTURE_NAMES):
    files = sorted(glob.glob(os.path.join(RAW_DIR, f"{gname}_*.npy")))
    test_files = files[int(0.8 * len(files)):]
    for fpath in test_files:
        raw = np.load(fpath)
        pred = classify_sequence(raw)
        all_preds.append(pred)
        all_true.append(g_idx)

all_preds = np.array(all_preds)
all_true = np.array(all_true)

valid_mask = all_preds != -1
mp_acc = np.mean(all_preds[valid_mask] == all_true[valid_mask])
coverage = valid_mask.mean()

print(f"MediaPipe rule-based accuracy: {mp_acc*100:.1f}%")
print(f"coverage: {coverage*100:.0f}% of samples got a prediction")

np.save("mp_predictions.npy", all_preds)
np.save("mp_labels.npy", all_true)

cm = confusion_matrix(all_true[valid_mask], all_preds[valid_mask])
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=GESTURE_NAMES).plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title(f"MediaPipe rule-based baseline  ({mp_acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("mp_cm.png", dpi=110)
plt.close()
print("saved mp_cm.png")
