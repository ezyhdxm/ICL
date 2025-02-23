import numpy as np
import pymc as pm
import aesara.tensor as at
import arviz as az

# --- Data Setup ---
# Assume N sequences, each of length T, and K possible states.
# (For illustration, we generate synthetic data.)
np.random.seed(42)
N = 10      # number of sequences
T = 20      # length of each sequence
K = 5       # number of states (indexed 0 to K-1)

true_P0 = np.array([np.random.dirichlet(np.ones(K)) for _ in range(K)])
true_P1 = np.array([np.random.dirichlet(np.ones(K)) for _ in range(K)])
true_transition_matrices = [true_P0, true_P1]

true_z = np.random.randint(0, 2, size=N)

observations = np.empty((N, T), dtype=int)

for i in range(N):
    z_i = true_z[i]
    observations[i, 0] = np.random.randint(0, K)
    for t in range(1, T):
        prev_state = observations[i, t-1]
        # sample next state according to row `prev_state` of the chosen transition matrix
        observations[i, t] = np.random.choice(K, p=true_transition_matrices[z_i][prev_state])

# --- Model Specification ---
with pm.Model() as model:
    # Prior over the latent regime for each sequence.
    # Here we assume both regimes are equally likely a priori.
    pi = pm.Dirichlet("pi", a=np.ones(2))
    z = pm.Categorical("z", p=pi, shape=N)

    # Transition matrices for the two regimes.
    # For each regime (2 total) and for each row (K rows) we draw from a Dirichlet prior.
    P = pm.Dirichlet("P", a=np.ones((2, K, K)), shape=(2, K, K))

    # Define a custom log-likelihood for a single sequence.
    # The probability of a sequence is:
    #   1/K * ∏_{t=2}^T P[z, s[t-1], s[t]]
    def logp_sequence(seq, z_idx, P_matrix):
        # Uniform probability for the first state: log(1/K)
        lp = -at.log(K)
        # Add the log-probability for transitions
        for t in range(1, T):
            prev = seq[t-1]
            curr = seq[t]
            lp = lp + at.log(P_matrix[z_idx, prev, curr])
        return lp
    
    # Compute the log-likelihood for each sequence and add it via a Potential.
    seq_logps = []
    for i in range(N):
        seq_logps.append(logp_sequence(observations[i], z[i], P))
    total_logp = at.sum(at.stack(seq_logps))
    pm.Potential("likelihood", total_logp)
    
    # --- Inference ---
    trace = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=42)

# --- Diagnostics ---
az.plot_trace(trace, var_names=["pi", "P"])
az.summary(trace, var_names=["pi", "P", "z"])