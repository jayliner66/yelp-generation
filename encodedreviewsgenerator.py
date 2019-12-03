import json
import numpy as np
from review_encoding import review_encoding
from review_encoding import words
from global_constants import N
####
def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

input_data = []
output_data = []
count = 0
for review in json_readline("yelp_dataset/review.json"):
    encoded_review = review_encoding(review["text"])
    input_data.append(encoded_review[:-1]) # START
    output_data.append(encoded_review[1:]) # STOP
    count += 1
    if(count == 100):
        break


with open('inputdata.json', 'w') as f:
    json.dump(input_data, f)
with open('outputdata.json', 'w') as f:
    json.dump(output_data, f)
