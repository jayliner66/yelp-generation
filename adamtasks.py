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
counter = 0
for review in json_readline("yelp_dataset/review.json"):
    counter += 1
    s = '!"#$%&()*+,-./:;?@[\\]^_`{|}~\t\n\r\x0b\x0c'
    words = review["text"]
    for i in range(len(s)):
        words = words.replace(s[i], " "+s[i]+" ")
    words = words.split(" ")
    for word in words:
        if(word == ""):
            continue
        word = word.lower()
        if(word in nums):
            word = nums[word]
        if word in count:
            count[word] += 1
        else:
            count[word] = 1
    if(counter % 1000 == 0):
        print(counter)
list_words = []
for i in count:
    list_words.append((-1*count[i], i))
list_words.sort()
with open('common_words.txt', 'w') as f:
    for i in range(10000):
        f.write(list_words[i][1] + "\n")
