#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:23:40 2019

@author: adam
"""
import json
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, GRU
from keras.layers.embeddings import Embedding

from review_encoding import words
from global_constants import batch_size, epochs, max_review_length

context_dim = 50
num_tokens = len(words)+1

BEAM_WIDTH = 10
max_length = max_review_length+1

START = words['START']
END = words['END']

context = Input(shape=(context_dim,))
decoder_input = Input(shape=(None, ))
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

###### TRAIN 

with open("inputdata.json") as f:
    input_data = np.asarray(json.load(f))
with open("outputdata.json") as f:
    output_data = np.asarray(json.load(f)).reshape(-1, max_length, 1)
with open("context.json") as f:
    context_array = np.asarray(json.load(f))

training_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
training_model.fit([context_array, input_data], output_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
##########

decoder_model.save('review_generation_model.h5')

