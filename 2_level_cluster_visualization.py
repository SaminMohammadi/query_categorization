import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
import collections
import matplotlib.pyplot as plt

def filter(text, sub_str):
    ret = [str.replace("d'","").replace("l'","").replace('"',"").replace("'","").replace(",","") for str in text if
            str not in sub_str] 
    return ret

def duplicate(input_list, n):
    new_list = [] 
    for i in range(n):
        new_list.extend(input_list)
    return new_list 

df = pd.read_csv("clustering_result_SG_2l.csv", sep = '\t', header = None, names = ["query","cluster","intent","sub_cluster"])
#print(df.head(5))
#i=0
#while i<= 5:
stopwords = ['un', 'une', 'le', 'les', 'la', 'de', 'de la', 'du', 'des', 'en', "l'",\
    'a', "a'", 'pour', 'à', 'au', 'on', 'et']

for cluster in df["cluster"].unique():
    df_new = df[df["cluster"]==cluster]
    tokens = df_new["query"].apply(word_tokenize)
    all_t = [" ".join(filter(text,stopwords)) for text in tokens]
    all_t = ' '.join(map(str,all_t))
    #tokens = [word.strip().replace("d'","").replace("l'","").replace('"',"").replace("'","").replace(",","") for word in str(all_t).split() if word not in stopwords]
    #tokens = list(tokens)
    #tokens = filter(lambda a: a not in ['bancaire', 'immobilier', 'compte', 'assurance'], tokens)
     #print(str(all_t[:50]))
    #all_t = ["".join(text) for text in tokens]
    wordcloud = WordCloud(stopwords=stopwords, collocations=False).generate(all_t)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(cluster)
    plt.axis("off")
    plt.show()
    
    filtered_words = [word for word in str(all_t).split()]
    counted_words = collections.Counter(filtered_words)

    words = []
    counts = []
    most_frequent_word = ''
    for letter, count in counted_words.most_common(3):
        if most_frequent_word=='':
            most_frequent_word = letter
        words.append(letter)
        counts.append(count)
    print (f"cluster {cluster}'s main vocabs:{words} with frenquncy of {counts}")
    df[df["cluster"]==cluster]['cluster_words'] = duplicate([words],df[df["cluster"]==cluster].shape[0])
    for sub_cluster in df_new["sub_cluster"].unique():
        df_sub = df_new[df_new["sub_cluster"]==sub_cluster]
        tokens = df_sub["query"].apply(word_tokenize)
        all_t = [" ".join(filter(text,stopwords)) for text in tokens]
        cluster_len = len(tokens)
        all_t = ' '.join(map(str,all_t))
        #tokens = df_sub["query"].apply(word_tokenize)# .replace(",",'').strip()
        #tokens = [word.strip().replace("'",'') for word in str(all_t).split() if word not in stopwords]
        #all_t = ["".join(text) for text in tokens]
        filtered_words = [word for word in str(all_t).split() if word not in [most_frequent_word]]
        counted_words = collections.Counter(filtered_words)

        words = []
        counts = []
        for letter, count in counted_words.most_common(3): 
            words.append(letter)
            counts.append(count)
        print (f"{cluster_len} :cluster {cluster}:sub-cluster {sub_cluster}'s main vocabs:{words} with frenquncy of {counts}")
        df[df[df["cluster"]==cluster]['sub_cluster']==sub_cluster]['sub_cluster_words'] = duplicate([words],df[df[df["cluster"]==cluster]['sub_cluster']==sub_cluster].shape[0])


