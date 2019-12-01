import json
import string
import nltk
from nltk import word_tokenize

words = {}

file = open("commonwords.txt", 'r')
i=1
for line in file:
    words[line[:-1]]=i
    i += 1

file.close()

text = ""
counter = 0

with open('yelp_dataset/review.json') as json_file:
    for line in json_file:
        text += json.loads(line)["text"]
        counter += 1
        if(counter > 10):
            break

def review_encoding(review):
    encoding = []
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    for token in tokens:
        if token in '!.?':
            encoding.append('END_SENT')
            #Any sentence stoppers, although may have errors, e.g. '.' may represent a decimal
        elif token in string.punctuation:
            encoding.append('PUNKT') #Any punctuation other than the above
        elif token.isnumeric():
            encoding.append('NUM') #Any integer, decimals will have already been split up
        elif token in words:
            encoding.append(words[token]) #Encoding of top 10000 words
        elif token.isalpha:
            encoding.append('RARE') #Any other words
        else:
            encoding.append('UNK') #Any unknown tokens, which there shouldn't be any
    return encoding

# print(review_encoding(text))
