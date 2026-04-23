# CS5100 Capstone — HMM Hand Gesture Recognition
**Northeastern University, Spring 2026**  
Based on Rabiner 1989 HMM Tutorial

---

## Project Overview

This project applies Hidden Markov Models to real-time hand gesture recognition using a webcam. Six gestures control a Pac-Man game. The HMM is trained using Baum-Welch and classifies using the Forward algorithm. Two baselines are implemented for comparison: DTW nearest neighbor (Sakoe and Chiba 1978) and a rule-based geometric classifier.

| Gesture | Action |
|---|---|
| Thumbs up | Move up |
| Point right | Move right |
| Point left | Move left |
| Thumbs down | Move down |
| Peace sign | Pause game (hold about 1.5 seconds) |
| Open palm | Reset position |

---

## Files in This Project

| File | Description |
|---|---|
| `collect_gestures.py` | Records webcam gesture data using MediaPipe |
| `features.py` | Converts raw landmarks to 38-dim feature vectors |
| `train_hmms.py` | Trains 6 GaussianHMMs using Baum-Welch via hmmlearn |
| `forward_algorithm.py` | Implements Forward algorithm from scratch (no library) |
| `demo.py` | Live Pac-Man game controlled by gesture recognition |
| `dtw_baseline.py` | DTW 1-nearest-neighbor baseline (Sakoe and Chiba 1978) |
| `mediapipe_baseline.py` | Rule-based geometric classifier baseline |
| `run_experiments.py` | Runs all three systems and prints full results |
| `analysis.py` | Loads saved results and prints detailed analysis |
| `compare_systems.py` | Generates comparison bar charts |
| `weather_verify.py` | Verifies Forward algorithm against Rabiner ice cream example |
| `latency.py` | Measures HMM inference latency in milliseconds |

Output charts are in the `outputs/` folder.

---

## How to Run — Complete Pipeline in Order

**Install dependencies first:**
```
pip install mediapipe hmmlearn opencv-python numpy matplotlib scikit-learn tslearn
```

**Step 1 — Collect gesture data**
```
python collect_gestures.py
```
Press 1 to 6 to record each gesture, q to quit. Target: 80 recordings per gesture, 6 gestures = 480 total.

**Step 2 — Extract features**
```
python features.py
```
Input: gesture_data_raw/ folder. Output: sequences.npy, labels.npy, feature_distributions.png

**Step 3 — Verify the HMM math**
```
python weather_verify.py
```
Expected output: P(O given model) = 1.536e-4 from both brute force and Forward algorithm.

**Step 4 — Train the HMM models**
```
python train_hmms.py
```
Output: hmm_models.pkl, hmm_cm.png, convergence.png, n_states_sweep.png

**Step 5 — Run DTW baseline**
```
python dtw_baseline.py
```
Note: this takes several minutes.

**Step 6 — Run rule-based baseline**
```
python mediapipe_baseline.py
```

**Step 7 — Run Forward algorithm from scratch**
```
python forward_algorithm.py
```
Prints verification that our implementation matches hmmlearn exactly.

**Step 8 — See all results**
```
python run_experiments.py
python analysis.py
```

**Step 9 — Generate comparison charts**
```
python compare_systems.py
```

**Step 10 — Run live demo**
```
python demo.py
```
Needs: hmm_models.pkl, hand_landmarker.task

---

## Key Design Decisions and Why

**Feature representation (38 dimensions per frame)**
- 15 joint bend angles using dot product geometry and arccos
- 4 finger spread angles between adjacent fingers
- 19 velocity features as frame-to-frame angle differences
- Angles used instead of raw coordinates because angles do not change when the hand moves in the frame

**HMM architecture**
- One GaussianHMM per gesture = 6 models total
- Left right topology: states advance only forward in time matching how a gesture physically unfolds
- Diagonal Gaussian: reduces parameters from 38x38=1444 to 38 per state, prevents overfitting
- 6 hidden states: matches expected gesture phases
- 10 random restarts: Baum-Welch finds local not global optima
- Confidence margin threshold of 5: gap between best and second-best log-likelihood must exceed 5 before acting

**Classification**
- Forward algorithm computes log P(O given lambda) per model
- Argmax over 6 models gives the predicted gesture
- Viterbi not used because we only need which gesture, not the hidden state sequence

---

## Dataset

| Property | Value |
|---|---|
| Recordings per gesture | 80 |
| Total gestures | 6 |
| Total sequences | 480 |
| Frames per sequence | 20 |
| Train per gesture | 64 (80 percent) |
| Test per gesture | 16 (20 percent) |
| Total test sequences | 96 |

---

## Results Summary

| System | Accuracy | Errors |
|---|---|---|
| HMM (Gaussian, left right, Baum-Welch) | 99.0% | 1 of 96 |
| DTW 1-NN (Sakoe and Chiba 1978) | 94.8% | 5 of 96 |
| Rule-based geometric classifier | 67.7% | approx. 31 of 96 |

HMM inference time: under 1ms per classification at 30 plus FPS  
DTW inference time: several minutes for full test set

---

## Known Limitations

- Data collected by one person only, not tested across users
- Angle features are translation invariant but not fully rotation invariant
- No rejection class for when no gesture is being made
- Left and right are mirror gestures distinguished mainly by velocity direction

---

## References

Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. Proceedings of the IEEE, 77(2), 257-286.

Sakoe, H. and Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43-49.