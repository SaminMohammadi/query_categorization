import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
#from nltk.tokenize import word_tokenize
### tokenize ###
import spacy
### stopwords ###
from spacy.lang.en.stop_words import STOP_WORDS

import collections
import matplotlib.pyplot as plt

import argparse 


parser = argparse.ArgumentParser("input arguments")
parser.add_argument("--corpus_name", default="fr_core_news_sm",\
     type=str, help="insert the pretrained statistical corpus name, default is for French, fr_core_news_sm")
args = parser.parse_args()

nlp = spacy.load(args.corpus_name)

df = pd.read_csv("../clustering_result_SG_2l.csv", sep = '\t', header = None, names = ["query","cluster","intent","sub_cluster","cluster_words","subcluster_words"])

for cluster in df["cluster"].unique():
    df_new = df[df["cluster"]==cluster]
    
    ### Tokenize and filter stop-words using nltk
    #tokens = df_new["query"].apply(word_tokenize)
    #all_t = [" ".join(filter(text,stopwords)) for text in tokens]
    
    ### Tokeniz, lemmatize and remove stop-words using Spacy 
    tokens = df_new["query"].apply(nlp)
    tokens =  map(lambda text: map(lambda x: x.lemma_, text), tokens)
    all_t = [" ".join(map(lambda x: str(x) if not nlp.vocab[str(x)].is_stop else "", text)) for text in tokens]
    all_t = ' '.join(map(str,all_t))

    wordcloud = WordCloud(collocations=False).generate(all_t)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(cluster)
    plt.axis("off")
    plt.show()
    

    filtered_words = [word for word in str(all_t).split()]
    counted_words = collections.Counter(filtered_words)

    cluster_words = []
    counts = []
    nc_kw = 2
    nsc_kw = 2 
    cluster_most_frequent_word = ''
    for letter, count in counted_words.most_common(nc_kw):
        if cluster_most_frequent_word=='':
            cluster_most_frequent_word = letter
        cluster_words.append(letter)
        counts.append(count)
    print (f"cluster {cluster}'s main vocabs:{cluster_words} with frenquncy of {counts}")
    
    df.at[df["cluster"]==cluster , 'cluster_words'] = ' '.join(map(str,cluster_words))
    print(df[df["cluster"]==cluster]['cluster_words'].notnull().unique())
    for sub_cluster in df_new["sub_cluster"].unique():
        df_sub = df_new[df_new["sub_cluster"]==sub_cluster]
        tokens = df_sub["query"].apply(nlp)
        tokens =  map(lambda text: map(lambda x: x.lemma_, text), tokens)
    
        all_t = [" ".join(map(lambda x: str(x) if not nlp.vocab[str(x)].is_stop else "", text)) for text in tokens]
        cluster_len = len(df_sub["query"])
        all_t = ' '.join(map(str,all_t))
        filtered_words = [word for word in str(all_t).split() if word not in cluster_words]
        counted_words = collections.Counter(filtered_words)

        words = []
        counts = []
        for letter, count in counted_words.most_common(nsc_kw): 
            words.append(letter)
            counts.append(count)
        print (f"{cluster_len} :cluster {cluster}:sub-cluster {sub_cluster}'s main vocabs:{words} with frenquncy of {counts}")
        r = df[(df["cluster"]==cluster) & (df['sub_cluster']==sub_cluster)].shape[0]
        df.at[(df["cluster"]==cluster) & (df['sub_cluster']==sub_cluster),'subcluster_words'] = ' '.join(map(str,words)) # duplicate([words],df_sub.shape[0])
df.to_csv("../named_clusters.csv")

