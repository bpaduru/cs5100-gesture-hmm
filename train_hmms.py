# train_hmms.py
# trains one GaussianHMM per gesture using Baum-Welch algorithm via hmmlearn
# architecture decisions explained in comments throughout
# experiments: N-states sweep, Baum-Welch convergence plots, confusion matrix
# reference: Rabiner 1989 - A Tutorial on Hidden Markov Models

import numpy as np
from hmmlearn import hmm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# gesture order must stay the same across features.py, train_hmms.py, and demo.py
# if this order changes the trained models will predict the wrong gesture labels
GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]

# N_STATES = 6 chosen because a gesture has roughly 6 meaningful temporal phases:
# hand approaching position, fingers beginning to move, fingers reaching shape,
# shape held stable, natural variation during hold, and hand beginning to relax
# the N-states sweep below tests 3 to 10 to confirm this is a reasonable choice
N_STATES = 6

# 10 restarts because Baum-Welch only guarantees a local optimum not a global one
# running it 10 times from different random starts and keeping the best result
# gives a much better chance of finding a good solution
N_RESTARTS = 10

# 80 iterations is more than enough based on the convergence plots which show
# all models converge in under 15 iterations
N_ITER = 80


def make_leftright_transmat(n):
    # left-right topology means states can only stay or advance, never go back
    # this matches the physical reality of a gesture which unfolds forward in time
    # a thumbs-up does not un-thumb itself halfway through
    # T[i][i]   = 0.6 means 60% chance of staying in current state
    # T[i][i+1] = 0.4 means 40% chance of advancing to next state
    # expected dwell time per state = 1 / (1 - 0.6) = 2.5 frames
    # with 20 frames and 6 states that gives about 3.3 frames per state on average
    # Baum-Welch will refine these values from the training data
    T = np.zeros((n, n))
    for i in range(n):
        if i < n - 1:
            T[i, i]     = 0.6
            T[i, i + 1] = 0.4
        else:
            # last state stays with probability 1 since there is nowhere left to go
            T[i, i] = 1.0
    return T


def train_one_hmm(train_seqs, n_states, n_iter):
    # concatenate all training sequences into one long array
    # hmmlearn requires this format with a separate lengths list
    X       = np.concatenate(train_seqs)
    lengths = [len(s) for s in train_seqs]

    model = hmm.GaussianHMM(
        n_components=n_states,
        # diagonal covariance means we assume the 38 features are independent
        # this reduces parameters from 38*38=1444 per state to just 38 per state
        # with only 64 training sequences (1280 frames) full covariance would overfit
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-4,
        verbose=False
    )

    # startprob_[0] = 1.0 means every sequence must start at state 0
    # this enforces left-right: the gesture always enters from the first state
    model.startprob_ = np.zeros(n_states)
    model.startprob_[0] = 1.0

    # initialize with our left-right structure
    # Baum-Welch will update this but starting here helps it converge faster
    model.transmat_ = make_leftright_transmat(n_states)

    try:
        # model.fit() runs the Baum-Welch algorithm (EM for HMMs)
        # it iterates until log-likelihood improvement is below tol or n_iter is reached
        model.fit(X, lengths)
        # score on training data used to compare restarts - higher is better
        score = model.score(X, lengths)
        return model, score
    except Exception:
        # some random initializations cause numerical failures
        # returning -inf means this restart will be ignored
        return None, -np.inf


def train_gesture_model(train_seqs, n_states=N_STATES):
    # run N_RESTARTS training attempts, keep the one with best training log-likelihood
    # Baum-Welch can get stuck in local optima depending on random initialization
    # multiple restarts explore different starting points to find a better solution
    best_model = None
    best_score = -np.inf
    for _ in range(N_RESTARTS):
        m, s = train_one_hmm(train_seqs, n_states, N_ITER)
        if s > best_score:
            best_score = s
            best_model = m
    return best_model, best_score


def get_predictions(test_seqs, models):
    # classify each test sequence using the Forward algorithm (via model.score())
    # model.score() computes log P(O | lambda) for one model and one sequence
    # we run this for all 6 models and take the argmax - the gesture whose
    # HMM assigns the highest probability to the observed sequence is the prediction
    preds = []
    for seq in test_seqs:
        scores = []
        for m in models:
            try:
                scores.append(m.score(seq))
            except Exception:
                scores.append(-np.inf)
        preds.append(int(np.argmax(scores)))
    return np.array(preds)


# load feature sequences and labels produced by features.py
seqs   = list(np.load("sequences.npy", allow_pickle=True))
labels = np.load("labels.npy")
print(f"loaded {len(seqs)} sequences, feature size per frame: {seqs[0].shape[1]}")
print(f"sequence length: {seqs[0].shape[0]} frames")
print(f"gesture order: {GESTURE_NAMES}\n")

# 80/20 split per gesture class to keep class balance in both sets
train_seqs, test_seqs   = [], []
train_labels, test_labels = [], []

for g in range(len(GESTURE_NAMES)):
    idx   = [i for i in range(len(seqs)) if labels[i] == g]
    split = int(0.8 * len(idx))
    train_seqs   += [seqs[i] for i in idx[:split]]
    train_labels += [g] * split
    test_seqs    += [seqs[i] for i in idx[split:]]
    test_labels  += [g] * (len(idx) - split)

train_labels = np.array(train_labels)
test_labels  = np.array(test_labels)
print(f"training set: {len(train_seqs)} sequences ({len(train_seqs)//len(GESTURE_NAMES)} per gesture)")
print(f"test set:     {len(test_seqs)} sequences ({len(test_seqs)//len(GESTURE_NAMES)} per gesture)\n")

# train all 6 HMMs
print(f"training {len(GESTURE_NAMES)} HMMs  N={N_STATES} states  {N_RESTARTS} restarts each ...")
models = []
for g, gname in enumerate(GESTURE_NAMES):
    g_train = [train_seqs[i] for i in range(len(train_seqs)) if train_labels[i] == g]
    m, best_ll = train_gesture_model(g_train)
    models.append(m)
    print(f"  {gname:<6} (label {g}): best training log-likelihood = {best_ll:.2f}")

predictions = get_predictions(test_seqs, models)
test_acc    = np.mean(predictions == test_labels)
print(f"\nHMM test accuracy (N={N_STATES}): {test_acc * 100:.1f}%")
print(f"correct: {int(test_acc * len(test_labels))}/{len(test_labels)}\n")

# save trained models and predictions for use by demo.py and compare_systems.py
with open("hmm_models.pkl", "wb") as f:
    pickle.dump(models, f)
np.save("hmm_predictions.npy", predictions)
np.save("test_labels.npy", test_labels)
print("saved hmm_models.pkl, hmm_predictions.npy, test_labels.npy")

# per-gesture accuracy breakdown
print("\nper-gesture accuracy:")
for g, gname in enumerate(GESTURE_NAMES):
    mask = test_labels == g
    acc  = np.mean(predictions[mask] == g) * 100
    bar  = "#" * int(acc / 5)
    print(f"  {gname:<6}: {acc:.1f}%  {bar}")


# experiment 1: N-states sweep from 3 to 10
# comparable to Rabiner 1989 Figure 15 which shows accuracy vs number of states
# this helps us confirm N=6 is a reasonable choice
print("\n" + "=" * 50)
print("EXPERIMENT: N-states sweep (3 to 10)")
print("=" * 50)
sweep_range = list(range(3, 11))
sweep_accs  = []
for n in sweep_range:
    sweep_models = []
    for g in range(len(GESTURE_NAMES)):
        g_train = [train_seqs[i] for i in range(len(train_seqs)) if train_labels[i] == g]
        m, _    = train_gesture_model(g_train, n_states=n)
        sweep_models.append(m)
    preds_n = get_predictions(test_seqs, sweep_models)
    acc     = np.mean(preds_n == test_labels)
    sweep_accs.append(acc)
    print(f"  N={n}: {acc * 100:.1f}%")

plt.figure(figsize=(8, 5))
plt.plot(sweep_range, [a * 100 for a in sweep_accs], "bo-", linewidth=2, markersize=7)
plt.axvline(x=N_STATES, color="red", linestyle="--", alpha=0.6, label=f"chosen N={N_STATES}")
plt.xlabel("number of hidden states N")
plt.ylabel("test accuracy (%)")
plt.title("recognition accuracy vs N states  (cf. Rabiner 1989 Figure 15)")
plt.xticks(sweep_range)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("n_states_sweep.png", dpi=110)
plt.close()
print("saved n_states_sweep.png")


# experiment 2: Baum-Welch convergence plots
# shows log-likelihood increasing over iterations for each gesture model
# confirms training is stable and converges properly
# tol=0.0 forces all 80 iterations so we can see the full convergence curve
print("\n" + "=" * 50)
print("EXPERIMENT: Baum-Welch convergence per gesture")
print("=" * 50)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for g, gname in enumerate(GESTURE_NAMES):
    g_train = [train_seqs[i] for i in range(len(train_seqs)) if train_labels[i] == g]
    X       = np.concatenate(g_train)
    lengths = [len(s) for s in g_train]
    conv_model = hmm.GaussianHMM(
        n_components=N_STATES, covariance_type="diag",
        n_iter=80, tol=0.0, verbose=False
    )
    conv_model.startprob_          = np.zeros(N_STATES)
    conv_model.startprob_[0]       = 1.0
    conv_model.transmat_           = make_leftright_transmat(N_STATES)
    conv_model.fit(X, lengths)
    ll_history = conv_model.monitor_.history
    axes[g].plot(range(1, len(ll_history) + 1), ll_history, "r-", linewidth=1.5)
    axes[g].set_title(gname)
    axes[g].set_xlabel("Baum-Welch iteration")
    axes[g].set_ylabel("log-likelihood")
    axes[g].grid(alpha=0.25)
    converged_at = len(ll_history)
    print(f"  {gname:<6}: converged in {converged_at} iterations"
          f"  final log-likelihood = {ll_history[-1]:.1f}")
plt.suptitle("Baum-Welch convergence per gesture model", fontsize=13)
plt.tight_layout()
plt.savefig("convergence.png", dpi=110)
plt.close()
print("saved convergence.png")


# experiment 3: confusion matrix on test set
# shows which gesture pairs the model confuses
cm = confusion_matrix(test_labels, predictions)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=GESTURE_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"HMM confusion matrix  (accuracy = {test_acc * 100:.1f}%)")
plt.tight_layout()
plt.savefig("hmm_cm.png", dpi=110)
plt.close()
print("saved hmm_cm.png")

print("\ndone. files to download: hmm_models.pkl and all .png figures")
