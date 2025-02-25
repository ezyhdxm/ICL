# The following code is adapted from https://github.com/JiajunSong629/ood-generalization-via-composition/blob/main/synthetic-experiments/data.py

import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import warnings
import os
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from tasks.utils import gen_tran_mat, calc_opt_err

warnings.simplefilter("ignore")

np.random.seed(2023)
torch.manual_seed(2023)


#####################################################
##################### Data generation ####################
#####################################################


def gen_simple_data(
    vocab,
    max_seq_len,
    sample_size,
    pattern="random",
    pattern_sample_len=None,
    rep_l=11,
    rep_h=20,
    return_lens=False,
):
    """
    Generate input sequences for training/testing based on different patterns.
    Simple repetitions of certain short-ranged patterns.
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        pattern: a string that indicated the short pattern used for generating sequence data, or a 1-d numpy array
        pattern_sample_len: the length of sampled patterns, only used when pattern ='random'
        return_lens: if True, returns repetition length for each seq
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long; if return_len, returns (data, lens)
    """
    vocab_size = vocab.size(0)
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    lens = np.zeros(sample_size, dtype=int)
    id0, id1, id2 = 0, 1, 2
    if max_seq_len % 12 != 0:
        warnings.warn("max_seq_len is not divisible by 12, which may cause issues!")

    for i in range(sample_size):
        if pattern == "random":
            if pattern_sample_len is None:
                pattern_len = np.random.randint(low=rep_l, high=rep_h)
            pattern_sample = torch.multinomial(
                torch.ones(vocab_size) / vocab_size, pattern_len, replacement=True
            )
            num_repeat = max_seq_len // pattern_len + 1
            r = pattern_len * num_repeat - max_seq_len
            tmp = vocab[pattern_sample].repeat(num_repeat)
            start_pos = np.random.randint(low=0, high=r)
            data[i, :] = tmp[start_pos : (start_pos + max_seq_len)]
            lens[i] = pattern_len
        else:  # for a given pattern
            pattern_len = len(pattern)
            num_repeat = max_seq_len // pattern_len + 1
            r = pattern_len * num_repeat - max_seq_len
            tmp = vocab[pattern].repeat(num_repeat)
            start_pos = np.random.randint(low=0, high=r)
            data[i, :] = tmp[start_pos : (start_pos + max_seq_len)]
            # warnings.warn('Pattern argument may not receive a correct input!')
            lens[i] = pattern_len

    data = (data, lens) if return_lens else data
    return data


def gen_repetition_data(
    vocab,
    max_seq_len,
    sample_size,
    distr=None,
    pattern_pool_size=None,
    patterns=None,
    rep_l=11,
    rep_h=20,
    num_repeat=2,
    return_lens=False,
):

    vocab_size = vocab.size(0)
    p = torch.ones(vocab_size) / vocab_size if distr is None else distr

    data = torch.multinomial(p, sample_size * max_seq_len, replacement=True).view(
        sample_size, max_seq_len
    )
    lens = np.zeros(sample_size, dtype=int)
    starts = np.zeros((sample_size, num_repeat), dtype=int)

    if pattern_pool_size is not None and patterns is None:
        # given the size of pattern pool, sample patterns from distribution p with length uniformly drawn from rep_l and repl_h
        patterns = []
        pattern_len_all = np.random.randint(
            low=rep_l, high=rep_h, size=pattern_pool_size
        )
        for t in range(pattern_pool_size):
            pattern = torch.multinomial(p, pattern_len_all[t], replacement=True)
            patterns.append(pattern)

    for i in range(sample_size):
        if pattern_pool_size is None and patterns is None:
            pattern_len = np.random.randint(low=rep_l, high=rep_h)
            pattern_sample = torch.multinomial(p, pattern_len, replacement=True)
        else:
            pattern_sample = patterns[np.random.randint(low=0, high=pattern_pool_size)]
            pattern_len = len(pattern_sample)

        r = max_seq_len - pattern_len * num_repeat
        gaps = torch.multinomial(torch.ones(r) / r, num_repeat, replacement=False)
        gaps = torch.sort(gaps)[0]
        gaps = torch.cat(
            (
                gaps[:1],
                torch.tensor([gaps[i] - gaps[i - 1] for i in range(1, num_repeat)]),
            )
        )
        start_pos = 0
        for j in range(num_repeat):
            start_pos = start_pos + gaps[j]
            data[i, start_pos : (start_pos + pattern_len)] = pattern_sample
            starts[i, j] = start_pos
            start_pos = start_pos + pattern_len
        lens[i] = pattern_len

    data = (data, lens, starts, patterns) if return_lens else data
    return data


def gen_simple_Aa_data(
    vocab,
    max_seq_len,
    sample_size,
    pattern=None,
    pattern_sample_len=None,
    rep_l=11,
    rep_h=20,
    return_lens=False,
):
    """
    Generate simple repetitions of certain short-ranged patterns. Each character has two versions (i.e., capitalization or not), sampled randomly
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        pattern: a string that indicated the short pattern used for generating sequence data, or a 1-d numpy array; if None, a random pattern will be sampled
        pattern_sample_len: the length of sampled patterns, only used when pattern ='random'
        return_lens: if True, returns repetition length for each seq
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long; if return_len, returns (data, lens)
    """
    vocab_size = vocab.size(0)
    vocab_halfsize = vocab_size // 2
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    lens = np.zeros(sample_size, dtype=int)

    for i in range(sample_size):
        if pattern is None:
            if pattern_sample_len is None:
                pattern_len = np.random.randint(low=rep_l, high=rep_h)
            pattern_sample = torch.multinomial(
                torch.ones(vocab_halfsize) / vocab_halfsize,
                pattern_len,
                replacement=True,
            )
            num_repeat = (max_seq_len - 1) // pattern_len + 1
            r = pattern_len * num_repeat - max_seq_len
            tmp = torch.zeros(num_repeat * pattern_len)
            for j in range(num_repeat):
                is_upper_case = torch.bernoulli(torch.tensor([0.5])).long()
                tmp[(j * pattern_len) : ((j + 1) * pattern_len)] = vocab[
                    pattern_sample + is_upper_case * vocab_halfsize
                ]
            start_pos = np.random.randint(low=0, high=r + 1)
            data[i, :] = tmp[start_pos : (start_pos + max_seq_len)]
            lens[i] = pattern_len
        else:  # for a given pattern
            pattern_len = len(pattern)
            num_repeat = (max_seq_len - 1) // pattern_len + 1
            r = pattern_len * num_repeat - max_seq_len
            tmp = torch.zeros(num_repeat * pattern_len)
            for j in range(num_repeat):
                is_upper_case = torch.bernoulli(torch.tensor([0.5])).long()
                tmp[(j * pattern_len) : ((j + 1) * pattern_len)] = vocab[
                    pattern + is_upper_case * vocab_halfsize
                ]
            start_pos = np.random.randint(low=0, high=r + 1)
            data[i, :] = tmp[start_pos : (start_pos + max_seq_len)]
            lens[i] = pattern_len

    data = (data, lens) if return_lens else data
    return data


def gen_hmm_data(
    vocab,
    max_seq_len,
    sample_size,
    state_sizes,
    transition_mat=None,
    sig=1.5,
    ioi=False,
    special_tokens=None,
    return_states=True,
    return_tokens_rep=False,
):
    """
    Generate simple HMM data: (z_1,z_2,...,z_T) is a latent Markov chain with z_t in {1,2,...,K}
    For each latent state z_t, observation is uniformly sampled from a set O_{z_t}
    The set O_1, O_2, ... O_K are non-overlapping and their cardinality is given by state_sizes
    sum_k state_sizes[k] must be no larger than vocab size.
    When ioi is True, we either sample two tokens [A] [B] from O_1 (if special_tokens is None) or choose special_tokens,
    and then set observables from O_1 to just repetition [A] [B] [A] [B] ...
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        state_sizes: a list/numpy array of integers indicating the size of sets for observables
        transition_mat: a K-by-K transition matrix for the latent Markov chain; if none, a tran_mat will be generated
        sig: the parameter used for generating a transition matrix if transition_mat is not provided
        ioi: if True, use ioi the scheme to sample two tokens from O_1 and repeat; otherwise uniformly sample tokens from O_1 independently
        special_tokens: a 1d torch array of length 2. If not None, use the two tokens for generating IOI
        return_states: if True, returns latent states
        return_tokens_rep: if True, returns tokens that are being repeated for each seq
    Returns:
        data: a list containing variables for the HMM,
        including input sequences, 2d torch.Tensor of type torch.long;
        transition_mat (useful when it is being generated in the function);
        if return_states is True, the latent states for each seq;
        if return_tokens_rep is True, for IOI also return the tokens being repeated
    """
    # check input arguments
    vocab_size = vocab.size(0)
    K = len(state_sizes)
    K_total = np.sum(state_sizes)
    size_cum = np.concatenate(([0], np.cumsum(state_sizes)))
    assert (
        np.all(state_sizes > 0) and K_total <= vocab_size
    ), "Wrong input for state_sizes"
    if transition_mat is not None:
        m1, m2 = transition_mat.shape
        assert (m1 == m2) and (
            m1 == K
        ), "Incorrect input dimension of transition matrix"
        assert torch.all(transition_mat >= 0) and torch.all(
            torch.abs(transition_mat.sum(dim=1) - 1) < 1e-6
        ), "Incorrect input of transition matrix"
    else:
        transition_mat = gen_tran_mat(K, 1, sig=sig)
    _, pi = calc_opt_err(transition_mat)  # get equilibrium distribution pi
    pi = torch.Tensor(pi).float()
    if return_tokens_rep:
        assert ioi, "ioi should be set to True"

    states = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    tokens_rep = torch.zeros(sample_size, 2).type(torch.LongTensor)
    states[:, 0] = torch.multinomial(pi, sample_size, replacement=True)
    for i in range(sample_size):
        size = state_sizes[states[i, 0]]
        data[i, 0] = size_cum[states[i, 0]] + torch.multinomial(
            torch.ones(size) / size, 1
        )
        for j in range(max_seq_len - 1):
            states[i, j + 1] = torch.multinomial(transition_mat[states[i, j], :], 1)
            size = state_sizes[states[i, j + 1]]
            data[i, j + 1] = size_cum[states[i, j + 1]] + torch.multinomial(
                torch.ones(size) / size, 1
            )
        if ioi:
            loc = states[i] == 0  # identify the state 0 for inserting repetition
            to_repeat_len = (loc.sum() + 1) // 2
            if special_tokens is None:
                tokens = torch.multinomial(
                    torch.ones(state_sizes[0]) / state_sizes[0],
                    to_repeat_len,
                    replacement=True,
                )  # sample a sequence to repeat
            else:
                tokens_idx = torch.multinomial(
                    torch.ones(len(special_tokens)) / len(special_tokens),
                    to_repeat_len,
                    replacement=True,
                )
                tokens = special_tokens[tokens_idx]
            data[i, loc] = tokens.repeat(2)[: loc.sum()]
            if return_tokens_rep:  # NOTE: buggy, set this to False
                tokens_rep[i] = tokens

    out = [data, transition_mat]
    if return_states:
        out.append(states)
    if return_tokens_rep:
        out.append(tokens_rep)
    return out


def gen_insert_data(
    vocab,
    max_seq_len,
    sample_size,
    background="random",
    background_random_weight=None,
    pattern="aaa",
    insrt_num_tokens=5,
    insrt_sep=1,
):
    """
    Generate input sequences for training/testing based on different patterns.
    Insert a short pattern in a purely random sequence (if background='random') or all-zero sequence
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        background: 'random' produces random sequence background, otherwise all-zero sequences
        background_random_weight: probability weight for generating random background, default is uniform random
        pattern: 'aaa' produces simple repetition pattern, 'random' produces a randomly sampled short pattern
                1d torch.Tensor plants the specified pattern into background, otherwise do nothing
        insrt_num_tokens: the number of tokens being planted
        insrt_sep: the index difference between consecutive tokens that are planted
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
        pattern: the torch.Tensor pattern being sampled if pattern='aaa' or 'random'
        pos_arr: 2d torch.Tensor, the indices of tokens planted in the sequences

    """
    assert type(insrt_num_tokens) == int, "insrt_num_tokens must be an odd integer"
    assert insrt_num_tokens % 2 == 1, "insrt_num_tokens must be an odd integer"
    vocab_size = vocab.size(0)
    k = insrt_num_tokens // 2
    if background_random_weight is None:  # uniform random noise in background
        background_random_weight = torch.ones(vocab_size) / vocab_size

    if background == "random":
        data = torch.multinomial(
            background_random_weight.repeat(sample_size, 1),
            max_seq_len,
            replacement=True,
        )  # random background
    else:
        data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)

    pos_arr = torch.zeros(sample_size, insrt_num_tokens)

    for i in range(sample_size):
        insrt_pos_center = torch.randint(
            k * insrt_sep, max_seq_len - k * insrt_sep, size=(1,)
        )
        insrt_pos = torch.arange(-k, k + 1) * insrt_sep + insrt_pos_center
        insrt_pos.type(torch.LongTensor)
        pos_arr[i, :] = insrt_pos
        if pattern == "aaa":
            pattern = torch.multinomial(torch.ones(vocab_size), 1).repeat(
                insrt_num_tokens
            )
            data[i, insrt_pos] = vocab[pattern]  # plant a simple repetition patter
        elif pattern == "random":
            pattern = torch.multinomial(
                torch.ones(vocab_size), insrt_num_tokens, replacement=True
            )  # a random pattern
            data[i, insrt_pos] = vocab[pattern]  # planted the pattern sampled earlier
        elif torch.is_tensor(pattern):
            data[i, insrt_pos] = vocab[
                pattern
            ]  # planted the pattern given by the argument
        else:
            pass  # do not plant signal, only has background

    return data, pattern, pos_arr