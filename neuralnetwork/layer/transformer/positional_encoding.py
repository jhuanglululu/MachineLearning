import math

import numpy as np

def encode_position(vectors):
    seq_len = len(vectors)
    d_model = len(vectors[0])
    position_vector = np.zeros((seq_len, d_model))
    for word in range(seq_len):
        for index in range(d_model):
            if index % 2 == 0:
                position_vector[word][index] = even_pe(d_model, word, index)
            else:
                position_vector[word][index] = odd_pe(d_model, word, index)

def even_pe(d_model, pos, index):
    return math.sin(pos / 10000 ** (index / d_model))

def odd_pe(d_model, pos, index):
    return math.cos(pos / 10000 ** (index / d_model))
