import os 
import pickle
import requests
import numpy as np
from util import *


def get_data(__file__, data_name="tinyshakespeare"):
    input_file_path = os.path.join(os.path.dirname(__file__), f'{data_name}.txt')
    if not os.path.exists(input_file_path):
        if data_name == 'tinyshakespeare':
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)
        else:
            raise ValueError(f'Unknown data_name: {data_name}')
        
    with open(input_file_path, 'r') as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    itos = get_itos(chars)
    stoi = get_stoi(chars)
    
    unigrams = get_unigrams(data)
    bigrams = get_bigrams(data)

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'unigrams': unigrams,
        'bigrams': bigrams,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)