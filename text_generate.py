#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:49:26 2019

@author: adam
"""
import json
import numpy as np

from keras.models import Model, load_model
from review_context import model

from review_encoding import words
from global_constants import batch_size, epochs, max_review_length

context_dim = 50
num_tokens = len(words)+1
k = 2000 #top k sampling

BEAM_WIDTH = 10
max_length = max_review_length+1

START = words['START']
END = words['END']

decoder_model = load_model('review_generation_model_100000.h5')

def sample_from(distribution_array, num_of_samples):
    # return list of indices, values corresponding to num_of_samples values in distribution_array (num_of_samples=1 would be argmax)
    a = np.argpartition(distribution_array, k)[:k]
    total_p = 0
    for i in a:
        total_p+=distribution_array[i]
    b = [distribution_array[i]/total_p for i in a]
    
    c = np.random.choice(a, size=num_of_samples, p=b)
    return [(i, distribution_array[i]) for i in c]

def generate_text(input_context):
    possibilities = [[[], START, input_context, 0]] #array of [sentence so far, current word, current hidden state, -log probability of sentence so far]
    not_stopped = True
    while not_stopped:
        new_possibilities = []
        for possibility in possibilities:
            if possibility[1]!=END:
                next_word_dist, hidden_state = decoder_model.predict([np.asarray(possibility[2]).reshape(1,context_dim), np.asarray([possibility[1]])])
                next_word_dist=next_word_dist[0][0]
                hidden_state=hidden_state
                #next_words = sample_from(next_word_dist, BEAM_WIDTH)
                next_words = np.argpartition(-1*np.asarray(next_word_dist), k-1)[:k]
                total_p=0
                for i in next_words:
                    total_p+=next_word_dist[i]
                weights = [next_word_dist[i]/total_p for i in next_words]
                next_words = np.random.choice(next_words, size=BEAM_WIDTH, p=weights)
                for next_word in next_words:
                    probability = next_word_dist[next_word]
                    new_possibilities+=[[possibility[0]+[next_word], next_word, hidden_state, possibility[3]-np.log(probability)]]
            else:
                new_possibilities+=[possibility]

        a = [tup[3] for tup in new_possibilities]
        b = np.argpartition(a, min(BEAM_WIDTH, len(new_possibilities))-1)
        possibilities = [new_possibilities[i] for i in b[:min(BEAM_WIDTH, len(new_possibilities))]] #BEAM_WIDTH lowest -log prob of new_possibilities

        continueloop = False
        for p in possibilities:
            if((len(p[0]) < max_length) and (p[1] != END)):
                continueloop = True
        if not continueloop:
            not_stopped = False
    return possibilities[0][0]


num_to_word = {num: word for word, num in words.items()}
num_to_word[0] = 'NOTHING'

def tokens_to_string(list_of_tokens):
    s = ''
    for token in list_of_tokens:
        s+=num_to_word[token]+' '
    return s

s1 = "seafood"

con = model[s1]

print(tokens_to_string(generate_text(con )))
