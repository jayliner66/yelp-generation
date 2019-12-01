import json
####
def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

count = {}
nums = {}
nums["0"] = "zero"
nums["1"] = "one"
nums["2"] = "two"
nums["3"] = "three"
nums["4"] = "four"
nums["5"] = "five"
nums["6"] = "six"
nums["7"] = "seven"
nums["8"] = "eight"
nums["9"] = "nine"
for review in json_readline("yelp_dataset/reviews.txt"):
    words = review["text"].split(" ")
    for word in words:
        if(word in nums):
            word = nums[word]
        if word in count:
            count[word] += 1
        else:
            count[word] = 1
list_words = []
for i in words:
    list_words.append((count[i], i))
sort(list_words)
with open('common_words.txt', 'w') as f:
    for i in range(10000):
        f.write(list_words[i][1] + "\n")
