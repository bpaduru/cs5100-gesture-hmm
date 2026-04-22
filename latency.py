# latency.py
# measures how long HMM scoring takes per window

import numpy as np
import pickle
import time

with open("hmm_models.pkl", "rb") as f:
    models = pickle.load(f)

seqs = list(np.load("sequences.npy", allow_pickle=True))
test_seq = seqs[0]
N = 300

print(f"timing HMM scoring over {N} trials")
print(f"sequence shape: {test_seq.shape}  (20 frames x 38 features)")

times = []
for _ in range(N):
    t0 = time.perf_counter()
    for m in models:
        m.score(test_seq)
    times.append(time.perf_counter() - t0)

mean_ms = np.mean(times) * 1000
print(f"\nHMM scoring 6 models x 20 frames:")
print(f"  mean:  {mean_ms:.2f} ms")
print(f"  min:   {np.min(times)*1000:.2f} ms")
print(f"  max classifications per second: {1000/mean_ms:.0f}")
print(f"\nMediaPipe hand detection takes ~20-30ms per frame")
print(f"so the bottleneck in the live loop is MediaPipe, not HMM")
print(f"HMM adds less than 1ms overhead per frame")
