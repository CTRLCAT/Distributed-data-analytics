import sys

current_word = None
word = None

predecesors = []
word_rank=[]

lim=10


# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    word, predecesor = line.split()

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        #Add predecesor to current list
        predecesors.append(predecesor)
    else:
        if current_word:
            if len(set(predecesors))>=lim:
                #Save results
                word_rank.append([current_word, len(set(predecesors))])
            #Restart predecesors
            predecesors = []
            predecesors.append(predecesor)
        #Update word
        current_word = word

if len(set(predecesors))>=lim:
    #Save results
    word_rank.append([current_word, len(set(predecesors))])

#Sort output
def sort_words(word_rank):
    word_rank.sort(key = lambda x: x[1], reverse=True)
    return word_rank

word_rank=sort_words(word_rank)

for word in word_rank:
    #Write sorted result to STDOUT
    print(f'{word[0]} {word[1]}')
