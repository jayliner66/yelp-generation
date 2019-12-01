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

def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

word_to_num = {}
num_to_word = {}
counter = 0
for word in json_readline("common_words.txt"):
    word_to_num[word] = counter
    num_to_word[counter] = word
    counter += 1

def review_to_num(review):
    nums = []
    s = '!"#$%&()*+,-./:;?@[\\]^_`{|}~\t\n\r\x0b\x0c'
    words = review
    for i in range(len(s)):
        words = words.replace(s[i], " "+s[i]+" ")
    words = words.split(" ")
    for w in words:
        if(w == " "):
            continue
        if w in word_to_num:
            nums.append(word_to_num[w])
        else:
            nums.append(10000) # other
    return nums

def sample_from(distribution_array, num_of_samples):
    # return list of indices, values corresponding to num_of_samples highest values in distribution_array (num_of_samples=1 would be argmax)
    new_dist_array = []
    for i in range(len(distribution_array)):
        new_dist_array.append((-1*distribution_array[i], i))
    new_dist_array.sort()
    ans = []
    for i in range(num_of_samples):
        ans.append(distribution_array[i][0])
    return ans

def generate_text(input_context):
    possibilities = [([], START, input_context, 0)] #array of [sentence so far, current word, current hidden state, -log probability of sentence so far]
    while not_stopped:#TODO: not_stopped based on END token or max length
        new_possibilities = []
        for possibility in possibilities:
            if possibility[1]!=END:
                next_word_dist, hidden_state = decoder_model.predict(possibility[2], possibility[1])
                next_words = sample_from(next_word_dist, BEAM_WIDTH)
                for next_word, probability in next_words:
                    new_possibilities+=[(possibility[0]+[next_word], next_word, hidden_state, possibility[3]-log(probability))]
            else:
                new_possibilities+=possibility
        new_possibilities.sort((key=lambda tup: tup[3], reverse=True)
        possibilities = new_possibilities[0:min(BEAM_WIDTH,len(new_possibilities))] #BEAM_WIDTH lowest -log prob of new_possibilities

    return possibilities[0][0]

    #TODO: return argmax -log probability of possibilities
