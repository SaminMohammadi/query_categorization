import csv
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#for item in vs1[0][:]:
 #   print(item)


import numpy as np
#For clustering 
import pandas as pd

df = pd.read_csv("../data/bank_SG.csv")
query_list = df["query"].tolist()





bc = BertClient(ip='localhost')
BERT_embedding = bc.encode(query_list)


km = KMeans()
km = KMeans(n_clusters=7)
km.fit(BERT_embedding)
clusters = km.labels_.tolist()

df_query_cluster = pd.DataFrame({'query':query_list,
                                'cluster': clusters,
                                'intent': df["Intention"].tolist(),
                                'sub_cluster':np.nan })
#df_query_cluster['sub_cluster'] = np.nan                        
for i in set(clusters):
    cluster_query_list = df_query_cluster[df_query_cluster['cluster'] == i]['query']
    cluster_BERT_embedding = bc.encode(cluster_query_list.to_list())
    km = KMeans(n_clusters=10)
    km.fit(cluster_BERT_embedding)
    #visualizer = KElbowVisualizer(km, k=(2,7))
    #visualizer.fit(cluster_BERT_embedding)
    #visualizer.show()
    clusters = km.labels_.tolist()
    print ('clusters:',len(clusters))
    print ('query:',len(cluster_query_list))
    df_query_cluster.at[df_query_cluster['cluster'] == i,'sub_cluster'] = clusters
# for 2-20 the best number of clueters id 7,8 
# and for 2-40 the best is 18
print(df_query_cluster.shape)
df_query_cluster.to_csv("../clustering_result_SG_2l.csv", sep = '\t', index = False, header = False)



