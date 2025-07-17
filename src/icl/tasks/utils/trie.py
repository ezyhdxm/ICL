class TrieNode:
    def __init__(self) -> None:
        self.children = {} # key: -1 or 1
        self.count = 0 # number of sequences that pass through this node
        self.count_pos = 0 # number of sequences where next token is 1
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, sequence):
        node = self.root
        for i in range(len(sequence) - 1):
            token = sequence[i]
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.count += 1
            if sequence[i+1] == 1:
                node.count_pos += 1