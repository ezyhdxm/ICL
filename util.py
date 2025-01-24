from collections import Counter, defaultdict

def get_unigrams(s):
    return Counter(s)

def get_bigrams(s):
    return Counter(zip(s, s[1:]))

def get_bigrams_cond(s):
    bigrams = get_bigrams(s)
    bigrams_cond = defaultdict(set)
    for (x, y), c in bigrams.items():
        bigrams_cond[x].add((y, c))
    return bigrams_cond

def get_itos(chars):
    return {i: c for i, c in enumerate(chars)}

def get_stoi(chars):
    return {c: i for i, c in enumerate(chars)} 