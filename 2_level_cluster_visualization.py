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
    str_list = ' '.join(map(str,input_list))
    #for i in range(n):
    #    new_list.append(str_list)
    #return new_list 
    return str_list
df = pd.read_csv("../clustering_result_SG_2l.csv", sep = '\t', header = None, names = ["query","cluster","intent","sub_cluster","cluster_words","subcluster_words"])
#print(df.head(5))
#i=0
#while i<= 5:
stopwords = ['un', 'une', 'le', 'les', 'la', 'de', 'de la', 'du', 'des', 'en', "l'",\
    'a', "a'", 'pour', 'à', 'au', 'on', 'et']

       

#df['cluster_words'] = np.nan
#df['subcluster_words'] = np.nan
for cluster in df["cluster"].unique():
    df_new = df[df["cluster"]==cluster]
    tokens = df_new["query"].apply(word_tokenize)
    all_t = [" ".join(filter(text,stopwords)) for text in tokens]
    all_t = ' '.join(map(str,all_t))
    wordcloud = WordCloud(stopwords=stopwords, collocations=False).generate(all_t)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(cluster)
    plt.axis("off")
    plt.show()
    
    filtered_words = [word for word in str(all_t).split()]
    counted_words = collections.Counter(filtered_words)

    cluster_words = []
    counts = []
    cluster_most_frequent_word = ''
    for letter, count in counted_words.most_common(3):
        if cluster_most_frequent_word=='':
            cluster_most_frequent_word = letter
        cluster_words.append(letter)
        counts.append(count)
    print (f"cluster {cluster}'s main vocabs:{cluster_words} with frenquncy of {counts}")
    #to_save = duplicate(words,df_new.shape[0])
    #to_ = np.reshape(to_save,[len(to_save),1])
    #print (len(to_save))
    #print (np.reshape(to_save,[len(to_save),1]).T)
    df.at[df["cluster"]==cluster , 'cluster_words'] = ' '.join(map(str,cluster_words))
    print(df[df["cluster"]==cluster]['cluster_words'].notnull().unique())
    for sub_cluster in df_new["sub_cluster"].unique():
        df_sub = df_new[df_new["sub_cluster"]==sub_cluster]
        tokens = df_sub["query"].apply(word_tokenize)
        all_t = [" ".join(filter(text,stopwords)) for text in tokens]
        cluster_len = len(tokens)
        all_t = ' '.join(map(str,all_t))
        filtered_words = [word for word in str(all_t).split() if word not in cluster_words]
        counted_words = collections.Counter(filtered_words)

        words = []
        counts = []
        for letter, count in counted_words.most_common(3): 
            words.append(letter)
            counts.append(count)
        print (f"{cluster_len} :cluster {cluster}:sub-cluster {sub_cluster}'s main vocabs:{words} with frenquncy of {counts}")
        r = df[(df["cluster"]==cluster) & (df['sub_cluster']==sub_cluster)].shape[0]
        df.at[(df["cluster"]==cluster) & (df['sub_cluster']==sub_cluster),'subcluster_words'] = ' '.join(map(str,words)) # duplicate([words],df_sub.shape[0])
df.to_csv("../named_clusters.csv")


