import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class DataArgs:
    k: int = 0
    seq_length: int = 256
    show_latents: bool = False
    fixed_special_toks: bool = False
    special_toks_offset: int = 0
    output_counter: bool = False
    no_repeat: bool = False
    vocab_size: int = 20

###################################
# Modified Original Implemetation #
###################################

class Dataset:
    def __init__(self, args: DataArgs,
                 train_test: Optional[str] = None,
                 bigram_outs: Optional[bool] = True):
        self.k = args.k
        self.seq_length = args.seq_length
        self.show_latents = args.show_latents
        self.train_test = train_test
        self.output_counter = args.output_counter
        self.no_repeat = args.no_repeat
        self.vocab_size = args.vocab_size
        self.bigram_outs = bigram_outs

        self.num_tokens = self.vocab_size
        self.tok_range = list(np.arange(self.num_tokens))

        self.n_train_toks = self.num_tokens
        self.marginal = np.ones(self.num_tokens) / self.num_tokens
        self.cond = [np.ones(self.num_tokens) / self.num_tokens for _ in range(self.num_tokens)]

        # special tokens
        self.idxs = None


    def gen_seq(self, rng: np.random.Generator):
        # select special tokens for this sequence
        idxs = list(rng.choice(self.tok_range, p=self.marginal, size=self.k, replace=False))
        
        if self.no_repeat:  # prevent next token to be same as idx
            pools = [self.tok_range.copy() for idx in idxs]
            for i, idx in enumerate(idxs):
                pools[i].remove(idx)
        else:
            pools = [self.tok_range for idx in idxs]
            
        if self.bigram_outs:
            outs = [rng.choice(pool, p=(self.cond[idx][pool] / self.cond[idx][pool].sum())) for pool, idx in zip(pools, idxs)]
        else:
            outs = [rng.choice(pool) for pool in pools]

        # print([(idxs[i], outs[i]) for i in range(self.k)])
        
        cnts = {}

        if self.show_latents:
            seq = idxs.copy()
            outputs_seq = [-1] * len(idxs) #  []
        else:
            seq = []
            outputs_seq = []
        seq += [rng.choice(self.tok_range, p=self.marginal)]
        while len(seq) < self.seq_length + 1:
            last = seq[-1]
            if last in idxs:
                seq.append(outs[idxs.index(last)])
                if self.output_counter:
                    cnts[last] = cnts.get(last, 0) + 1
                    outputs_seq.append(cnts[last])
                else:
                    outputs_seq.append(1)
            else:
                probs = self.cond[last]
                outputs_seq.append(0)
                seq.append(rng.choice(self.tok_range, p=probs))
        outputs_seq.append(0)

        return seq, outputs_seq

    def gen_seqs(self, rng: np.random.Generator) -> List[str]:
        while True:
            seq, outputs_seq = self.gen_seq(rng)
            yield (seq, outputs_seq)

    def gen_batch(self, rng: np.random.Generator, batch_size: int):
        seqs = []
        outs = []
        for _ in range(batch_size):
            seq, out = self.gen_seq(rng)
            seqs += seq
            outs += out
        x = np.array(seqs).reshape(batch_size, self.seq_length + 1)
        outs = np.array(outs).reshape(batch_size, self.seq_length + 1)
        return x, outs


def iterate_batches(dataset: Dataset,
                    batch_size: int = 20,
                    num_workers: int = 6,
                    seed: int = 42):
    def worker(queue, rng):
        while True:
            x, outs = dataset.gen_batch(rng, batch_size)
            queue.put((x, outs))

    import multiprocessing as mp
    q = mp.Queue(maxsize=1000)
    processes = [mp.Process(target=worker, args=(q, np.random.default_rng([seed, i]))) for i in range(num_workers)]
    for p in processes:
        p.start()

    seq = []
    outputs_seq = []
    count = 0
    try:
        while True:
            x, outs = q.get()
            yield (x[:,:-1], x[:,1:], outs[:,:-1])
    except:
        for p in processes:
            p.kill()