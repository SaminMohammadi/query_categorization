import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
import collections
import matplotlib.pyplot as plt
from bert_serving.client import BertClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

def filter(text, sub_str):
    ret = [str.replace("d'","").replace("l'","").replace('"',"").replace("'","").replace(",","") for str in text if
            str not in sub_str] 
    return ret

def duplicate(input_list, n):
    new_list = []
    str_list = ' '.join(map(str,input_list))
    return str_list
df = pd.read_csv("../clustering_result_SG_2l.csv", sep = '\t', header = None, names = ["query","cluster","intent","sub_cluster","cluster_words","subcluster_words"])
stopwords = ['un', 'une', 'le', 'les', 'la', 'de', 'de la', 'du', 'des', 'en', "l'",\
    'a', "a'", 'pour', 'à', 'au', 'on', 'et']

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
lda = LDA(n_components=5)
count_vectorizer = CountVectorizer()
count_data = count_vectorizer.fit_transform([df['query']])
output = lda.fit(count_data)

print_topics(lda, count_vectorizer, 1)

tokens = df["query"].apply(word_tokenize)
all_t = [" ".join(filter(text,stopwords)) for text in tokens]
all_t = ' '.join(map(str,all_t))
    
filtered_words = [word for word in str(all_t).split()]
counted_words = collections.Counter(filtered_words)

words = ['voiture', 'assurance', 'immibilier', 'pret', 'compte', 'bancaire', 'livret',\
    'bourse', 'credit', 'simulation']

query_list = df["query"].tolist()
bc = BertClient(ip='localhost')
bc.encode_async()
key_words_embedding = bc.encode(words)
for query in query_list:
    for word in query.split():
        word_vec = bc.encode([word])[0]
        score = np.sum(word_vec * key_words_embedding, axis=1) / np.linalg.norm(key_words_embedding, axis=1)
        topk_idx = np.argsort(score)[::-1][:2]
        print (query)
        for idx in topk_idx:
            print('> %s\t%s' % (score[idx], words[idx]))



