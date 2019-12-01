import json
import string
import nltk
import random
from nltk import word_tokenize
from nltk.corpus import stopwords

text = ""

count = {}
special_count = {}

counter = 0
#currently, the top 10000 most common words are chosen out of the reviews which are
#taken with 2% probability
with open('yelp_dataset/review.json') as json_file:
    for line in json_file:
        if random.random() < 0.02:
            text += json.loads(line)["text"]
        # text += json.loads(line)["text"]
        # counter += 1
        # if (counter >10):
        #     break

# print(counter)

tokens = word_tokenize(text)
tokens = [w.lower() for w in tokens]

table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
words = [word for word in stripped if word.isalpha()]
# rids words of punctuation and numerals

for word in words:
    count[word] = count.get(word, 0)+1

sorted_count = sorted(count.items(), key = lambda x: x[1], reverse=True)

file = open("commonwords.txt", 'w')
for i in range(10000):
    file.write(sorted_count[i][0]+'\n')
file.close()

stop_words = set(stopwords.words('english'))
special_words = [w for w in words if not w in stop_words]

for word in special_words:
    special_count[word] = special_count.get(word, 0)+1

sorted_special_count = sorted(special_count.items(), key = lambda x: x[1], reverse=True)
# print(sorted_special_count[:1000])
# only contains 'non-common' words

