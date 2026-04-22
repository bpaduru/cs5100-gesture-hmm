# compare_systems.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]

seqs = list(np.load("sequences.npy", allow_pickle=True))
labels = np.load("labels.npy")

test_seqs, test_labels = [], []
for g in range(len(GESTURE_NAMES)):
    idx = [i for i in range(len(seqs)) if labels[i] == g]
    split = int(0.8 * len(idx))
    test_seqs += [seqs[i] for i in idx[split:]]
    test_labels += [g] * (len(idx) - split)
test_labels = np.array(test_labels)

hmm_preds = np.load("hmm_predictions.npy")
dtw_preds = np.load("dtw_predictions.npy")
mp_preds  = np.load("mp_predictions.npy")
mp_true   = np.load("mp_labels.npy")

hmm_acc = np.mean(hmm_preds == test_labels)
dtw_acc = float(np.load("dtw_accuracy.npy")[0])
valid_mp = mp_preds != -1
mp_acc   = np.mean(mp_preds[valid_mp] == mp_true[valid_mp]) if valid_mp.sum() > 0 else 0.0

print("=" * 52)
print("FINAL RESULTS")
print("=" * 52)
print(f"HMM (Gaussian, left-right, Baum-Welch):  {hmm_acc*100:.1f}%")
print(f"DTW (1-NN, Sakoe and Chiba 1978):         {dtw_acc*100:.1f}%")
print(f"MediaPipe rule-based baseline:            {mp_acc*100:.1f}%")
print("=" * 52)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

systems = ["HMM\n(our method)", "DTW\n(Sakoe 1978)", "MediaPipe\nbaseline"]
accs    = [hmm_acc*100, dtw_acc*100, mp_acc*100]
colors  = ["#4878CF", "#E07B39", "#6ABF69"]

bars = axes[0].bar(systems, accs, color=colors, width=0.45, edgecolor="black", linewidth=0.7)
for bar, a in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{a:.1f}%", ha="center", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 108)
axes[0].set_ylabel("accuracy (%)")
axes[0].set_title("Overall accuracy on held-out test set")
axes[0].grid(axis="y", alpha=0.35)

per_g_hmm = [np.mean(hmm_preds[test_labels == g] == g) * 100 for g in range(len(GESTURE_NAMES))]
per_g_dtw = [np.mean(dtw_preds[test_labels == g] == g) * 100 for g in range(len(GESTURE_NAMES))]

x = np.arange(len(GESTURE_NAMES))
w = 0.35
axes[1].bar(x - w/2, per_g_hmm, w, label="HMM", color="#4878CF", edgecolor="black", linewidth=0.6)
axes[1].bar(x + w/2, per_g_dtw, w, label="DTW", color="#E07B39", edgecolor="black", linewidth=0.6)
axes[1].set_xticks(x)
axes[1].set_xticklabels(GESTURE_NAMES)
axes[1].set_ylabel("accuracy (%)")
axes[1].set_title("Per-gesture accuracy: HMM vs DTW")
axes[1].set_ylim(0, 110)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.35)

plt.tight_layout()
plt.savefig("comparison.png", dpi=110)
plt.close()
print("saved comparison.png")
