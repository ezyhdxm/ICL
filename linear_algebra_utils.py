import numpy as np

########################
# Linear Algebra Utils #
#########################

def effective_rank(A):
    # See https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-12))  # Add small epsilon to avoid log(0)
    return np.exp(entropy)

def stable_rank(A):
    frob_norm = np.linalg.norm(A, ord='fro')
    op_norm = np.linalg.norm(A, ord=2)  # spectral norm
    return (frob_norm ** 2) / (op_norm ** 2)