import json
import review_encoding
####
def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

input = []
output = []
count = 0
for review in json_readline("yelp_dataset/review.json"):
    arr = review_encoding(review)
    input.append([10015] + review) # START
    output.append(review + [10016]) # STOP
    count += 1
    if(count == 1000):
        break

with open('inputdata.json', 'w') as f:
    json.dump(input, f)
with open('outputdata.json', 'w') as f:
    json.dump(output, f)
