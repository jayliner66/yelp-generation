#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:23:40 2019

@author: adam
"""

from keras.models import Model
from keras.layers import Dense, Input, GRU
from keras.layers.embeddings import Embedding

context_dim = 50
num_tokens = 10000

BEAM_WIDTH = 10


context = Input(shape=(context_dim,))
decoder_input = Input(shape=(None,))
decoder_embed = Embedding(input_dim=num_tokens, output_dim=context_dim, mask_zero=True)

gru_1 = GRU(context_dim, return_sequences=True, return_state=False)
gru_2 = GRU(context_dim, return_sequences=True, return_state=True)

decoder_dense = Dense(num_tokens, activation='softmax')

embedded_word = decoder_embed(decoder_input)
gru_1_output = gru_1(embedded_word, initial_state=context)
gru_2_output, h = gru_2(gru_1_output)
output = decoder_dense(gru_2_output)

training_model = Model([context, decoder_input], output)

decoder_model = Model([context, decoder_input], [output, h])

def sample_from(distribution_array, num_of_samples):
    #TODO: return list of indices, values corresponding to num_of_samples highest values in distribution_array (num_of_samples=1 would be argmax)

def generate_text(input_context):
    possibilities = [[[], START, input_context, 0]] #array of [sentence so far, current word, current hidden state, -log probability of sentence so far]
    while not_stopped:#TODO: not_stopped based on END token or max length
        new_possibilities = []
        for possibility in possibilities:
            if possibility[1]!=END:
                next_word_dist, hidden_state = decoder_model.predict(possibility[2], possibility[1])
                next_words = sample_from(next_word_dist, BEAM_WIDTH)
                for next_word, probability in next_words:
                    new_possibilities+=[[possibility[0]+[next_word], next_word, hidden_state, possibility[3]-log(probability)]]
            else:
                new_possibilities+=possibility
        possibilities = #BEAM_WIDTH lowest -log prob of new_possibilities
    #TODO: return argmax -log probability of possibilities