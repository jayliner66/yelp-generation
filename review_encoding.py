import json
import string
import nltk
from nltk import word_tokenize

from global_constants import N, max_review_length


words = {}

file = open("commonwords.txt", 'r')
i=1
for line in file:
    words[line[:-1]]=i
    i += 1
    
assert len(words)==N

words['PUNKT']=N+1
words['DOLLAR']=N+2
words['NUM']=N+3
words['RARE']=N+4
words['UNK']=N+5
words['END_SENT']=N+6
words['START']=N+7
words['END']=N+8
file.close()



def review_encoding(review):
    encoding = []
    tokens = word_tokenize(review)
    tokens = [w.lower() for w in tokens]
    for token in tokens:
        if token in '!.?':
            encoding.append(words['END_SENT'])
            #Any sentence stoppers, although may have errors, e.g. '.' may represent a decimal
        elif token in '$':
            encoding.append(words['DOLLAR']) #Dollar sign
        elif token in string.punctuation:
            encoding.append(words['PUNKT']) #Any punctuation other than the above
        elif token.isnumeric():
            encoding.append(words['NUM']) #Any integer, decimals will have already been split up
        elif token in words:
            encoding.append(words[token]) #Encoding of top 10000 words
        elif token.isalpha():
            encoding.append(words['RARE']) #Any other words
        else:
            encoding.append(words['UNK']) #Any unknown tokens, which there shouldn't be any
    return [words['START']]+encoding[:max_review_length]+[0]*(max_review_length-len(encoding))+[words['END']]

