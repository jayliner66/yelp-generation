import json
import numpy as np
from review_encoding import review_encoding
from review_encoding import words
from review_context import make_context
from review_context import model
from global_constants import N
####
def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

input_data = []
output_data = []
context = []
count = 0
for review in json_readline("yelp_dataset/review.json"):
    encoded_review = review_encoding(review["text"])
    review_context = make_context(review["text"])
    input_data.append(encoded_review[:-1]) # START
    output_data.append(encoded_review[1:]) # STOP
    context.append(review_context)
    count += 1
    if(count % 100 == 0):
        print(count)


with open('inputdata.json', 'w') as f:
    json.dump(input_data, f)
with open('outputdata.json', 'w') as f:
    json.dump(output_data, f)
with open('context.json', 'w') as f:
    json.dump(context, f)
