###############################################
#
#   This python code clusters queries using KMeans and DBScan algorithms 
#   Since the results of KMeans are better we keep DBScan's code lines commneted
#   This code uses Bert_as_a_service, so to run this, you need to install and run it first.  
#           pip install bert-serving-server  
#           pip install bert-serving-client 
#   use googl's  pretrained version
#           bert-serving-start -model_dir=./bert-folder/multi_cased_L-12_H-768_A-12 
#   use your own fune-tuned version
#           bert-serving-start -model_dir= ./my_bert/multi_cased_L-12_H-768_A-12 \
#                   -tuned_model_dir= ./my_bert/bert_output/ -ckpt_name=model.ckpt-374
#   P.s. -ckpt_name comes from bert_output folder which is the name of model check point  
#   
############################################### 

import csv
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from yellowbrick.cluster import KElbowVisualizer
import argparse 


parser = argparse.ArgumentParser("input arguments")
parser.add_argument("--corpus_name",  default="fr_core_news_sm",\
     type=str, help="insert the pretrained statistical corpus name, default is for French, fr_core_news_sm")
args = parser.parse_args()

### tokenize ###
import spacy
nlp = spacy.load(args.corpus_name) 
### stopwords ###
from spacy.lang.en.stop_words import STOP_WORDS

import numpy as np
import pandas as pd

df = pd.read_csv("../data/bank_SG.csv")
query_list = df["query"].tolist()

no_clusters = 5
no_subclusters = 5

bc = BertClient(ip='localhost')

### Tokeniz, lemmatize and remove stop-words using Spacy 
tokens = df["query"].apply(nlp)
tokens =  map(lambda text: map(lambda x: x.lemma_, text), tokens)
query_list = [" ".join(map(lambda x: str(x) if not nlp.vocab[str(x)].is_stop else "", text)) for text in tokens]
    
BERT_embedding = bc.encode(query_list)


km = KMeans()
km = KMeans(n_clusters=no_clusters)
km.fit(BERT_embedding)
clusters = km.labels_.tolist()
'''
### testing the accuracy of DBScan
dbscan1 = DBSCAN(eps= 8, min_samples=4,).fit(BERT_embedding)
clusters = dbscan1.labels_.tolist()
'''
df_query_cluster = pd.DataFrame({'query':df["query"].tolist(),
                                'cluster': clusters,
                                'intent': df["Intention"].tolist(),
                                'sub_cluster':np.nan })
for i in set(clusters):
    cluster_query_list = df_query_cluster[df_query_cluster['cluster'] == i]['query']
    
    ### Tokeniz, lemmatize and remove stop-words using Spacy 
    tokens = cluster_query_list.apply(nlp)
    tokens =  map(lambda text: map(lambda x: x.lemma_, text), tokens)
    cluster_query_list = [" ".join(map(lambda x: str(x) if not nlp.vocab[str(x)].is_stop else "", text)) for text in tokens]
    
    
    cluster_BERT_embedding = bc.encode(cluster_query_list)
    
    km = KMeans(n_clusters=no_subclusters)
    km.fit(cluster_BERT_embedding)
    clusters = km.labels_.tolist()
    '''
    ### testing the accuracy of DBScan
    dbscan = DBSCAN(eps=8, min_samples=3).fit(cluster_BERT_embedding)
    clusters = dbscan.labels_.tolist()
    '''

    print ('clusters:',len(clusters))
    print ('query:',len(cluster_query_list))
    df_query_cluster.at[df_query_cluster['cluster'] == i,'sub_cluster'] = clusters

print(df_query_cluster.shape)
df_query_cluster.to_csv("../clustering_result_SG_2l.csv", sep = '\t', index = False, header = False)



