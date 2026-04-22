# weather_verify.py
# verifies the HMM forward algorithm matches brute force on Rabiner example
# expected result is P(O|model) = 1.536e-4

import numpy as np
from itertools import product
from hmmlearn import hmm

# Rabiner section II ice cream weather model
# state 0 = hot day, state 1 = cold day
# observation 0 = ate 1 ice cream, 1 = ate 2, 2 = ate 3
start_prob = np.array([0.6, 0.4])

trans_mat = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

emit_mat = np.array([
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]
])

obs = np.array([2, 0, 2, 2, 1, 1, 2, 0, 2])
T = len(obs)
n_states = 2

# brute force: sum over all 2^9 = 512 possible state sequences
brute_total = 0.0
for state_seq in product(range(n_states), repeat=T):
    p = start_prob[state_seq[0]] * emit_mat[state_seq[0]][obs[0]]
    for t in range(1, T):
        p *= trans_mat[state_seq[t-1]][state_seq[t]]
        p *= emit_mat[state_seq[t]][obs[t]]
    brute_total += p

print("brute force P(O|model):", f"{brute_total:.6e}")

# hmmlearn forward algorithm (same computation, O(N^2 * T) instead of O(N^T))
model = hmm.CategoricalHMM(n_components=2)
model.startprob_ = start_prob
model.transmat_ = trans_mat
model.emissionprob_ = emit_mat

log_p = model.score(obs.reshape(-1, 1))
forward_result = np.exp(log_p)
print("hmmlearn forward P(O|model):", f"{forward_result:.6e}")

if abs(brute_total - forward_result) < 1e-10:
    print("both match, hmmlearn forward algorithm is working correctly")
    print("ready to train gesture models")
else:
    print("mismatch, check setup before proceeding")

log_viterbi, best_states = model.decode(obs.reshape(-1, 1), algorithm="viterbi")
label_map = {0: "hot", 1: "cold"}
print("viterbi most likely states:", [label_map[s] for s in best_states])
