CS5100 Capstone - HMM Hand Gesture Recognition
Northeastern University, Spring 2026
Based on Rabiner 1989 HMM Tutorial

================================================================
PROJECT OVERVIEW
================================================================

This project applies Hidden Markov Models to real-time hand gesture
recognition using a webcam. Six gestures control a Pac-Man game.
The HMM is trained using Baum-Welch and classifies using the Forward
algorithm. Two baselines are implemented for comparison: DTW nearest
neighbor (Sakoe and Chiba 1978) and a rule-based geometric classifier.

Gestures:
  thumbs up    -> move up
  point right  -> move right
  point left   -> move left
  thumbs down  -> move down
  peace sign   -> pause game (hold about 1.5 seconds)
  open palm    -> reset position

================================================================
FILES IN THIS PROJECT
================================================================

collect_gestures.py   - records webcam gesture data using MediaPipe
features.py           - converts raw landmarks to 38-dim feature vectors
train_hmms.py         - trains 6 GaussianHMMs using Baum-Welch via hmmlearn
forward_algorithm.py  - implements Forward algorithm from scratch (no library)
demo.py               - live Pac-Man game controlled by gesture recognition
dtw_baseline.py       - DTW 1-nearest-neighbor baseline (Sakoe and Chiba 1978)
mediapipe_baseline.py - rule-based geometric classifier baseline
run_experiments.py    - runs all three systems and prints full results
analysis.py           - loads saved results and prints detailed analysis
compare_systems.py    - generates comparison bar charts
weather_verify.py     - verifies Forward algorithm against Rabiner ice cream example
latency.py            - measures HMM inference latency in milliseconds

================================================================
HOW TO RUN - COMPLETE PIPELINE IN ORDER
================================================================

step 1: collect gesture data (run on your laptop with webcam)
  python collect_gestures.py
  press 1-6 to record each gesture, q to quit
  target: 80 recordings per gesture, 6 gestures = 480 total

step 2: extract features
  python features.py
  input:  gesture_data_raw/ folder
  output: sequences.npy, labels.npy, feature_distributions.png

step 3: verify the HMM math (optional but recommended)
  python weather_verify.py
  confirms hmmlearn Forward algorithm matches brute force
  expected output: P(O given model) = 1.536e-4 from both methods

step 4: train the HMM models
  python train_hmms.py
  input:  sequences.npy, labels.npy
  output: hmm_models.pkl, hmm_predictions.npy,
          hmm_cm.png, convergence.png, n_states_sweep.png

step 5: run DTW baseline
  python dtw_baseline.py
  input:  sequences.npy, labels.npy
  output: dtw_predictions.npy, dtw_accuracy.npy, dtw_cm.png
  note:   this takes several minutes

step 6: run rule-based baseline
  python mediapipe_baseline.py
  input:  gesture_data_raw/ folder
  output: mp_predictions.npy, mp_labels.npy, mp_cm.png

step 7: run our Forward algorithm from scratch
  python forward_algorithm.py
  input:  hmm_models.pkl, sequences.npy, labels.npy
  output: printed verification and per-gesture accuracy

step 8: see all results and analysis
  python run_experiments.py    runs everything fresh
  python analysis.py           loads saved results and explains them

step 9: generate comparison charts
  python compare_systems.py
  output: comparison.png

step 10: run live demo
  python demo.py
  needs: hmm_models.pkl, hand_landmarker.task

================================================================
KEY DESIGN DECISIONS AND WHY
================================================================

feature representation (38 dimensions per frame):
  - 15 joint bend angles using dot product geometry and arccos
  - 4 finger spread angles between adjacent fingers
  - 19 velocity features as frame-to-frame angle differences
  - angles used instead of raw coordinates because angles do not
    change when the hand moves position in the frame

HMM architecture:
  - one GaussianHMM per gesture = 6 models total
  - left right topology: states advance only forward in time
    matching how a gesture physically unfolds
  - diagonal Gaussian: assumes 38 features are independent
    reduces parameters from 38x38=1444 to 38 per state
    prevents overfitting given the dataset size
  - 6 hidden states: matches expected gesture phases
  - 10 random restarts: Baum-Welch finds local not global optima
  - confidence margin threshold of 5: gap between best and
    second-best log-likelihood must exceed 5 before acting

classification:
  - Forward algorithm computes log P(O given lambda) per model
  - argmax over 6 models gives the predicted gesture
  - Viterbi not used because we only need which gesture,
    not the hidden state sequence

================================================================
DATASET
================================================================

  recordings per gesture : 80
  total gestures         : 6
  total sequences        : 480
  frames per sequence    : 20
  train per gesture      : 64  (80 percent split)
  test per gesture       : 16  (20 percent split)
  total test sequences   : 96

================================================================
RESULTS SUMMARY
================================================================

  HMM accuracy   : 99.0%  (95 of 96 test samples correct)
  DTW accuracy   : 94.8%  (91 of 96 test samples correct)
  Rule-based     : 67.7%  (fails especially on down and left)

  HMM inference time : under 1ms per classification (30 plus FPS)
  DTW inference time : several minutes for full test set

================================================================
KNOWN LIMITATIONS
================================================================

  - data collected by one person only, not tested across users
  - angle features are translation invariant but not fully
    rotation invariant, tilting hand changes values
  - no rejection class for when no gesture is being made
  - left and right are mirror gestures sharing similar angles,
    distinguished mainly by velocity direction

================================================================
REFERENCES
================================================================

  Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models and
  Selected Applications in Speech Recognition. Proceedings of
  the IEEE, 77(2), 257-286.

  Sakoe, H. and Chiba, S. (1978). Dynamic programming algorithm
  optimization for spoken word recognition. IEEE Transactions on
  Acoustics, Speech, and Signal Processing, 26(1), 43-49.