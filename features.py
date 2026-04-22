# features.py
# converts raw landmark files into feature vectors for HMM training
# creates sequences.npy and labels.npy 

import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# triplets for computing joint bend angles at each finger segment
# each entry is (parent_joint, joint, child_joint)
# angle is computed at the middle index
JOINT_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),       # thumb: CMC, MCP, IP
    (0,5,6),(5,6,7),(6,7,8),       # index: MCP, PIP, DIP
    (0,9,10),(9,10,11),(10,11,12), # middle: MCP, PIP, DIP
    (0,13,14),(13,14,15),(14,15,16),# ring: MCP, PIP, DIP
    (0,17,18),(17,18,19),(18,19,20) # pinky: MCP, PIP, DIP
]

# triplets for finger spread angles between finger bases
SPREAD_TRIPLETS = [
    (5,0,9),   # index to middle spread at wrist
    (9,0,13),  # middle to ring spread at wrist
    (13,0,17), # ring to pinky spread at wrist
    (1,0,5)    # thumb to index spread at wrist
]

# label 0=up, 1=right, 2=left, 3=down, 4=peace, 5=palm
GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]


def angle_at_joint(a, b, c):
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cos_val)


def frame_to_angles(landmarks):
    # landmarks shape (21, 3), returns 19 angles
    angles = []
    for a_i, b_i, c_i in JOINT_TRIPLETS:
        angles.append(angle_at_joint(landmarks[a_i], landmarks[b_i], landmarks[c_i]))
    for a_i, b_i, c_i in SPREAD_TRIPLETS:
        angles.append(angle_at_joint(landmarks[a_i], landmarks[b_i], landmarks[c_i]))
    return np.array(angles)


def sequence_to_features(raw_seq):
    # raw_seq shape (20, 21, 3)
    # returns (20, 38): 19 angle features + 19 velocity features per frame
    # velocity = first difference across frames, captures motion direction
    angle_seq = np.array([frame_to_angles(frame) for frame in raw_seq])
    velocity = np.diff(angle_seq, axis=0, prepend=angle_seq[:1])
    return np.concatenate([angle_seq, velocity], axis=1)


def load_all(raw_dir="gesture_data_raw"):
    sequences = []
    labels = []
    for label, gname in enumerate(GESTURE_NAMES):
        files = sorted(glob.glob(os.path.join(raw_dir, f"{gname}_*.npy")))
        print(f"  {gname} (label {label}): {len(files)} sequences")
        for path in files:
            raw = np.load(path)
            feat_seq = sequence_to_features(raw)
            sequences.append(feat_seq)
            labels.append(label)
    return sequences, np.array(labels)


def plot_distributions(sequences, labels, save_path="feature_distributions.png"):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    feature_labels = [
        "thumb MCP angle", "index MCP angle", "middle MCP angle",
        "index PIP angle", "thumb-index spread", "index-middle spread"
    ]
    feat_indices = [0, 3, 6, 4, 15, 16]
    for plot_i, feat_i in enumerate(feat_indices):
        ax = axes[plot_i]
        for label, gname in enumerate(GESTURE_NAMES):
            g_seqs = [sequences[j] for j in range(len(sequences)) if labels[j] == label]
            if g_seqs:
                vals = np.concatenate([s[:, feat_i] for s in g_seqs])
                ax.hist(vals, bins=25, alpha=0.45, label=gname, density=True)
        ax.set_title(feature_labels[plot_i], fontsize=10)
        ax.legend(fontsize=6)
        ax.set_xlabel("radians")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()
    print("saved", save_path)


if __name__ == "__main__":
    print("loading gesture sequences from gesture_data_raw/...")
    seqs, labels = load_all("gesture_data_raw")
    print(f"\ntotal sequences: {len(seqs)}")
    print(f"feature vector size per frame: {seqs[0].shape[1]}")
    print(f"sequence length: {seqs[0].shape[0]} frames")

    np.save("sequences.npy", np.array(seqs, dtype=object), allow_pickle=True)
    np.save("labels.npy", labels)
    print("\nsaved sequences.npy and labels.npy")
    print("upload both files to Google Colab to run train_hmms.py")

    plot_distributions(seqs, labels)
