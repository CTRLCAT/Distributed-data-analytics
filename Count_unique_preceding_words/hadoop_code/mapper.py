import sys

#Max distance of connections
link_range=1

#Puntuation
punctuation="!'’#$%&)*+,-./:;<=>?@[\]^_`{|}~"

#Stop words
stop_words=['i', 'me', 'my', 'myself', 'we', 'our',
             'ours', 'ourselves', 'you', "you're", "you've", 
             "you'll", "you'd", 'your', 'yours', 'yourself', 
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 
             "she's", 'her', 'hers', 'herself', 'it', "it's", 
             'its', 'itself', 'they', 'them', 'their', 'theirs', 
             'themselves', 'what', 'which', 'who', 'whom', 'this', 
             'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 
             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
             'an', 'the', 'and', 'but', 'if', 'or', 'because', 
             'as', 'until', 'while', 'of', 'at', 'by', 'for', 
             'with', 'about', 'against', 'between', 'into', 'through', 
             'during', 'before', 'after', 'above', 'below', 'to', 
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
             'under', 'again', 'further', 'then', 'once', 'here', 
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 
             'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
             'don', "don't", 'should', "should've", 'now', 'd', 'll', 
             'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
             'couldn', "couldn't", 'didn', "didn't", 'doesn', 
             "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
             "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
             'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
             "weren't", 'won', "won't", 'wouldn', "wouldn't", 'to', 'towards',
             'in', 'a', 'would', 'chapter']




#Input comes from STDIN (standard input)
for line in sys.stdin:
    if len(line)>0:
        #Initialize tuples in line
        tuples=[]

        #Delete space at beginning and end
        line=line.strip()

        #To lower case
        line=line.lower()

        for pun in punctuation:
            #Remove puntuation
            line=line.replace(pun,'')
        for word in stop_words:
            #Remove stop words
            replace_word=word+' '
            line=line.replace(replace_word,'')
            replace_word=' '+word
            line=line.replace(replace_word,'')

        #Separate words
        line=line.split()

        for each in line:
            each=each.encode('utf-8')
        
        for i in range(len(line)):
            for j in range(link_range):
                #If the index exists for both word and predecesor, add them as tuples
                try:
                    tuples.append([line[i-j-1],line[i]])
                except IndexError:
                    continue        

        for each_tuple in tuples:
            # write the results to STDOUT (standard output);
            # what we output here will be the input for the
            # Reduce step, i.e. the input for reducer.py
            #
            # Send node and whether there is a link between each one of the neighbors as 1 or 0
            print(f'{each_tuple[1]} {each_tuple[0]}')
