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
import argparse

parser = argparse.ArgumentParser("input arguments")
parser.add_argument("--corpus_name", default="fr_core_news_sm",\
     type=str, help="insert the pretrained statistical corpus name, default is for French, fr_core_news_sm")
args = parser.parse_args()

### tokenizer/lemmatizer ###
import spacy
nlp = spacy.load(args.corpus_name) 
### stopwords ###
from spacy.lang.en.stop_words import STOP_WORDS


def filter(text, sub_str):
    ret = [str.replace("d'","").replace("l'","").replace('"',"").replace("'","").replace(",","") for str in text if
            str not in sub_str] 
    return ret

def duplicate(input_list, n):
    new_list = []
    str_list = ' '.join(map(str,input_list))
    return str_list

df = pd.read_csv("../clustering_result_SG_2l.csv", sep = '\t', header = None, names = ["query","cluster","intent","sub_cluster","cluster_words","subcluster_words"])

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

tokens = df["query"].apply(nlp)
tokens =  map(lambda text: map(lambda x: x.lemma_, text), tokens)
query_list = [" ".join(map(lambda x: str(x) if not nlp.vocab[str(x)].is_stop else "", text)) for text in tokens]


lda = LDA(n_components=5)
count_vectorizer = CountVectorizer()
count_data = count_vectorizer.fit_transform(query_list) 
output = lda.fit(count_data)

print_topics(lda, count_vectorizer, 1)

all_t_lemma_stop = [" ".join(map(lambda x: str(x) if not nlp.vocab[str(x)].is_stop else "", text)) for text in tokens]
all_t = ' '.join(map(str,all_t_lemma_stop))
    
filtered_words = [word for word in str(all_t).split()]
counted_words = collections.Counter(filtered_words)

words = ['voiture', 'assurance', 'immibilier', 'pret', 'compte', 'bancaire', 'livret',\
    'bourse', 'credit', 'simulation']

bc = BertClient(ip='localhost')
key_words_embedding = bc.encode(words)
for query in query_list:
    for word in query.split():
        word_vec = bc.encode([word])[0]
        score = np.sum(word_vec * key_words_embedding, axis=1) / np.linalg.norm(key_words_embedding, axis=1)
        topk_idx = np.argsort(score)[::-1][:1]
        print (query)
        for idx in topk_idx:
            print('> %s\t%s' % (score[idx], words[idx]))



