import json
import string
import nltk
import random
import numpy as np
from random import randint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.data import find
from gensim.models import KeyedVectors

# from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file = 'glove.6B.50d.txt'
# word2vec_output_file = 'glove.6B.50d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

dim = 50

# counter = 0

# with open('yelp_dataset/review.json') as json_file:
#     for line in json_file:
#         if counter==0:
#             text = json.loads(line)["text"]
#             break
#         counter += 1

filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

wordlist = open("commonwords.txt", 'r')
word_to_context = {}

for line in wordlist:
    if line[:-1] in model:
        word_to_context[line[:-1]] = model[line[:-1]]
    else:
        word_to_context[line[:-1]] = model["um"]

def make_context(review):
    num_of_random_words = randint(1,3)
    tokens = word_tokenize(review)
    tokens = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    special_words = [w for w in words if not w in stop_words]

    if special_words == []:
        special_words = words
    if(special_words == []):
        special_words = ["nothing"]

    selected_words = []
    for i in range(num_of_random_words):
        while(True):
            word = random.choice(special_words)
            if word in word_to_context:
                selected_words.append(word)
                break


    total_context = np.zeros(dim)
    for i in range(0,num_of_random_words):
        total_context = np.add(total_context, word_to_context[selected_words[i]])

    return total_context.tolist()

# print(make_context(text))
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

# print(selected_words)
