#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:49:26 2019
@author: adam
"""
import json
import numpy as np

from keras.models import Model, load_model

from review_encoding import words
from global_constants import batch_size, epochs, max_review_length

context_dim = 50
num_tokens = len(words)+1

BEAM_WIDTH = 10
max_length = max_review_length+1
k=500

START = words['START']
END = words['END']

decoder_model = load_model('review_generation_model_300000.h5')

def sample_from(distribution_array, num_of_samples):
    # return list of indices, values corresponding to num_of_samples values in distribution_array (num_of_samples=1 would be argmax)
    u = np.argpartition(-1*np.asarray(distribution_array), k-1)[:k]


    total = 0
    for i in u:
        total += distribution_array[i]
    weights = np.asarray([distribution_array[i] for i in u]) * (1.0/total)

    a = np.random.choice(u, size=num_of_samples, p=weights)
    return [(i, distribution_array[i]) for i in a]

def generate_text(input_context):
    possibilities = [([], START, input_context, 0)] #array of [sentence so far, current word, current hidden state, -log probability of sentence so far]
    not_stopped = True
    while not_stopped:
        new_possibilities = []
        for possibility in possibilities:
            if possibility[1]!=END:
                next_word_dist, hidden_state = decoder_model.predict([np.asarray(possibility[2]).reshape(1,context_dim), np.asarray([possibility[1]])])
                next_word_dist=next_word_dist[0][0]
                hidden_state=hidden_state
                next_words = sample_from(next_word_dist, BEAM_WIDTH)
                for next_word, probability in next_words:
                    new_possibilities+=[(possibility[0]+[next_word], next_word, hidden_state, possibility[3]-np.log(probability))]
            else:
                new_possibilities+=[possibility]
        new_possibilities.sort(key=lambda tup: tup[3], reverse=True)
        possibilities = new_possibilities[0:min(BEAM_WIDTH,len(new_possibilities))] #BEAM_WIDTH lowest -log prob of new_possibilities

        continueloop = False
        for p in possibilities:
            if((len(p[0]) < max_length) and (p[1] != END)):
                continueloop = True
        if not continueloop:
            not_stopped = False
    return possibilities[0][0]


num_to_word = {num: word for word, num in words.items()}


def tokens_to_string(list_of_tokens):
    s = ''
    for token in list_of_tokens:
        s+=num_to_word[token]+' '
    return s

print(tokens_to_string(generate_text(np.zeros(50))))
