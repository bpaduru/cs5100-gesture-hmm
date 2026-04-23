# dtw_baseline.py
# DTW 1-nearest neighbor baseline from Sakoe and Chiba 1978


import numpy as np
from tslearn.metrics import dtw
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]

seqs = list(np.load("sequences.npy", allow_pickle=True))
labels = np.load("labels.npy")

train_seqs, test_seqs = [], []
train_labels, test_labels = [], []
for g in range(len(GESTURE_NAMES)):
    idx = [i for i in range(len(seqs)) if labels[i] == g]
    split = int(0.8 * len(idx))
    train_seqs += [seqs[i] for i in idx[:split]]
    train_labels += [g] * split
    test_seqs += [seqs[i] for i in idx[split:]]
    test_labels += [g] * (len(idx) - split)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


def dtw_classify(test_seq, train_seqs, train_labels):
    best_dist = np.inf
    best_label = -1
    for tr_seq, label in zip(train_seqs, train_labels):
        d = dtw(test_seq, tr_seq)
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label


print(f"running DTW 1-NN on {len(test_seqs)} test sequences...")
print("this takes a few minutes since it compares each test to all 384 training sequences")
t0 = time.time()
dtw_preds = []
for i, ts in enumerate(test_seqs):
    dtw_preds.append(dtw_classify(ts, train_seqs, train_labels))
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(test_seqs)}")

dtw_preds = np.array(dtw_preds)
dtw_acc = np.mean(dtw_preds == test_labels)
print(f"\nDTW accuracy: {dtw_acc*100:.1f}%  ({time.time()-t0:.0f}s)")

np.save("dtw_predictions.npy", dtw_preds)
np.save("dtw_accuracy.npy", np.array([dtw_acc]))

cm = confusion_matrix(test_labels, dtw_preds)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=GESTURE_NAMES).plot(ax=ax, cmap="Oranges", colorbar=False)
ax.set_title(f"DTW confusion matrix  (accuracy = {dtw_acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("dtw_cm.png", dpi=110)
plt.close()
print("saved dtw_cm.png")