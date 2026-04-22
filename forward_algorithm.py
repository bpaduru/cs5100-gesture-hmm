# forward_algorithm.py
# implements the HMM Forward algorithm from scratch for Gaussian HMMs
# this directly addresses Problem 1 from Rabiner 1989:
# given a model lambda and observation sequence O, compute P(O | lambda)
# written entirely without relying on hmmlearn for the actual computation
# at the end it verifies our result matches hmmlearn's model.score()
# to confirm correctness

import numpy as np
import pickle


def log_sum_exp(log_vals):
    # numerically stable way to compute log(sum(exp(x)))
    # if we just did sum(exp(x)) the values underflow to 0 for very negative log probs
    # so we factor out the maximum value first to keep things in a safe range
    max_val = np.max(log_vals)
    if max_val == -np.inf:
        return -np.inf
    return max_val + np.log(np.sum(np.exp(log_vals - max_val)))


def gaussian_log_prob(obs, mean, var):
    # log probability of a single observation vector under a diagonal Gaussian
    # obs  : shape (D,) - one frame of 38 features
    # mean : shape (D,) - the emission mean for this HMM state
    # var  : shape (D,) - the diagonal variances for this state
    # formula: log N(x; mu, sigma^2) =
    #   -0.5 * sum( (x - mu)^2 / sigma^2 ) - 0.5 * sum( log(2 * pi * sigma^2) )
    # the 1e-300 prevents log(0) when a variance is very small
    log_p  = -0.5 * np.sum((obs - mean) ** 2 / (var + 1e-300))
    log_p -= 0.5 * np.sum(np.log(2.0 * np.pi * (var + 1e-300)))
    return log_p


def forward_algorithm(obs_seq, startprob, transmat, means, vars):
    # full Forward algorithm in log space - Rabiner 1989 Section III
    # obs_seq   : shape (T, D) - T frames of D-dimensional feature vectors
    # startprob : shape (N,)   - pi, probability of starting in each state
    # transmat  : shape (N, N) - A, transition probability matrix
    # means     : shape (N, D) - emission mean per hidden state
    # vars      : shape (N, D) - diagonal emission variance per hidden state
    # returns log P(O | lambda) and the alpha matrix for inspection

    T = len(obs_seq)  # number of frames in the sequence
    N = len(startprob)  # number of hidden states

    # log_alpha[t][i] = log P(o_1, o_2, ..., o_t, q_t = i | lambda)
    # this is the joint probability of being in state i at time t
    # AND having seen all observations up to time t
    log_alpha = np.full((T, N), -np.inf)

    # step 1 - initialization at t = 0
    # for each state i: log_alpha[0][i] = log(pi[i]) + log(b_i(o_0))
    # where b_i(o) is the Gaussian emission probability of state i
    for i in range(N):
        if startprob[i] > 0:
            log_emit = gaussian_log_prob(obs_seq[0], means[i], vars[i])
            log_alpha[0][i] = np.log(startprob[i]) + log_emit

    # step 2 - recursion from t = 1 to T - 1
    # for each state j at time t, sum contributions from all states i at t-1
    # log_alpha[t][j] = log( sum_i( exp(log_alpha[t-1][i]) * A[i][j] ) ) + log(b_j(o_t))
    # in log space we use log_sum_exp to handle the sum over i
    for t in range(1, T):
        for j in range(N):
            log_emit = gaussian_log_prob(obs_seq[t], means[j], vars[j])
            # gather contributions from all previous states
            incoming = np.array([
                log_alpha[t - 1][i] + np.log(transmat[i][j] + 1e-300)
                for i in range(N)
            ])
            log_alpha[t][j] = log_sum_exp(incoming) + log_emit

    # step 3 - termination
    # P(O | lambda) = sum over all states at final time step T-1
    log_prob = log_sum_exp(log_alpha[T - 1])

    return log_prob, log_alpha


def classify_sequence(obs_seq, models):
    # classify a single gesture sequence using our Forward algorithm
    # runs the Forward algorithm against all 6 trained HMMs
    # returns the gesture index with the highest log P(O | lambda)
    log_scores = []
    for m in models:
        startprob = m.startprob_
        transmat  = m.transmat_
        means     = m.means_
        # hmmlearn stores diagonal covariance as (N, D) array of variances
        vars_     = m.covars_
        log_p, _  = forward_algorithm(obs_seq, startprob, transmat, means, vars_)
        log_scores.append(log_p)
    return int(np.argmax(log_scores)), np.array(log_scores)


if __name__ == "__main__":

    GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]

    print("=" * 60)
    print("HMM FORWARD ALGORITHM - FROM SCRATCH VERIFICATION")
    print("Rabiner 1989, Section III - Problem 1")
    print("=" * 60)

    print("\nloading trained HMM models ...")
    with open("hmm_models.pkl", "rb") as f:
        models = pickle.load(f)

    print("loading sequences and labels ...")
    seqs   = list(np.load("sequences.npy", allow_pickle=True))
    labels = np.load("labels.npy")

    # build same 80/20 test split as train_hmms.py
    test_seqs, test_labels = [], []
    for g in range(len(GESTURE_NAMES)):
        idx   = [i for i in range(len(seqs)) if labels[i] == g]
        split = int(0.8 * len(idx))
        test_seqs   += [seqs[i] for i in idx[split:]]
        test_labels += [g] * (len(idx) - split)
    test_labels = np.array(test_labels)

    print(f"\nrunning our Forward algorithm on {len(test_seqs)} test sequences ...")
    print("comparing against hmmlearn model.score() for each sequence\n")

    max_diff  = 0.0
    our_preds = []

    for idx, seq in enumerate(test_seqs):
        our_label, our_scores = classify_sequence(seq, models)

        # hmmlearn log-likelihood for the winning model
        hmm_log_p = models[our_label].score(seq)
        our_log_p = our_scores[our_label]
        diff = abs(our_log_p - hmm_log_p)
        max_diff = max(max_diff, diff)
        our_preds.append(our_label)

        if idx < 6:
            print(f"  seq {idx:02d}  gesture: {GESTURE_NAMES[test_labels[idx]]:<6}"
                  f"  our log-p: {our_log_p:.3f}"
                  f"  hmmlearn log-p: {hmm_log_p:.3f}"
                  f"  diff: {diff:.8f}")

    our_preds = np.array(our_preds)
    our_acc   = np.mean(our_preds == test_labels)

    print(f"\nmax difference between our Forward and hmmlearn: {max_diff:.8f}")
    print(f"accuracy using our from-scratch Forward algorithm: {our_acc * 100:.1f}%")

    if max_diff < 0.01:
        print("\nverification passed - our Forward algorithm matches hmmlearn exactly")
    else:
        print("\ncheck implementation - differences are larger than expected")

    print("\nper-gesture accuracy using our Forward algorithm:")
    for g, gname in enumerate(GESTURE_NAMES):
        mask = test_labels == g
        acc  = np.mean(our_preds[mask] == g) * 100
        bar  = "#" * int(acc / 5)
        print(f"  {gname:<6}: {acc:.1f}%  {bar}")
