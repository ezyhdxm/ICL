from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import sys

from typing import List, Optional, Tuple

@dataclass
class DataArgs:
    num_spe_toks: int = 0
    seq_len: int = 256
    file_path = 'meta.pkl'

class Dataset:
    def __init__(self, args: DataArgs):
        self.num_spe_toks = args.num_spe_toks
        self.seq_len = args.seq_len
        self.meta = pickle.load(open(args.file_path, 'rb'))
        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']
        self.vocab_size = self.meta['vocab_size']
        
        self.marginal = np.zeros(self.vocab_size)
        for w, cnt in self.meta['unigrams'].items():
            self.marginal[self.stoi[w]] = cnt
        self.marginal /= self.marginal.sum()

        self.cond = [np.zeros(self.vocab_size) for _ in range(self.vocab_size)]
        for (w1, w2), cnt in self.meta['bigrams'].items():
            self.cond[self.stoi[w1]][self.stoi[w2]] = cnt
        for w in range(self.vocab_size):
            self.cond[w] /= self.cond[w].sum()
        
    
    def get_sequence(self, rng: np.random.Generator):
        idxs = list(rng.choice(self.vocab_size, 
                               self.num_spe_toks, 
                               replace=False, p=self.marginal))
        outs = [rng.choice(self.vocab_size) for _ in idxs]
        seq = [rng.choice(self.vocab_size, p=self.marginal)]
        seq_mask = []
        while len(seq) < self.seq_len+1:
            last = seq[-1]
            if last in idxs:
                seq.append(outs[idxs.index(last)])
                seq_mask.append(1)
            else:
                probs = self.cond[last]
                seq.append(rng.choice(self.vocab_size, p=probs))
                seq_mask.append(0)
        seq_mask.append(0)
        return seq, seq_mask
    
    def get_seqs(self, rng: np.random.Generator):
        while True:
            seq, seq_mask = self.get_sequence(rng)
            yield (seq, seq_mask)
    
    def get_batch(self, rng: np.random.Generator, batch_size: int):
        seqs, masks = [], []
        for _ in range(batch_size):
            seq, mask = self.get_sequence(rng)
            seqs.append(seq)
            masks.append(mask)
        x = np.array(seqs).reshape(batch_size, -1)
        masks = np.array(masks).reshape(batch_size, -1)
        return x, masks
    

def iterate_batches(dataset: Dataset, 
                    batch_size: int, num_workers: int, seed: int):
    
    from multiprocessing import Process, Queue

    def worker(queue, rng):
        while True:
            queue.put(dataset.get_batch(rng, batch_size))

    q = Queue(maxsize=1000)
    processes = [
        Process(target=worker, args=(q, np.random.default_rng([seed, i])))
        for i in range(num_workers)
        ]
    
    for p in processes:
        p.start()
    
    try:
        while True:
            x, mask = q.get()
            yield (x[:,:-1], x[:,1:], mask[:,:-1])
    except:
        for p in processes:
            p.kill()

