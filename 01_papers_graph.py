import pandas as pd
import numpy as np
from functions.TextEmbedding import *
from functions.AnomaliesDetection import *
import pickle
from matplotlib.pyplot import step, show, plot
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from graphdatascience import GraphDataScience
from multiprocessing import Pool
import time
from tqdm import tqdm
import lance
import pyarrow as pa
import os
import shutil
from neo4j import GraphDatabase
import re
from sklearn.neighbors import NearestNeighbors
import sklearn


C_Optimizer = CFunctions()

URI = "bolt://localhost:17180"
AUTH = ("", "")

driver = GraphDatabase.driver(URI, auth=AUTH)

events = driver.execute_query("""
MATCH k=(e:Entity)<-[:AFFILIATED_WITH]-(a:Author)-[:AUTHORED]->(p:Paper)-[:IS_CLASSIFIED_AS]->(s:Subject)
RETURN DISTINCT p.title AS Title, p.embedding as PaperEmb, COLLECT(s.embedding) AS SubjectsEmb, COLLECT(e.embedding) AS EntitiesEmb, COLLECT(a.embedding) AS AuthorsEmb
""")
events = pd.DataFrame(events[0], columns = events[-1])

events['SubjectsEmbS'] = events['SubjectsEmb'].apply(lambda x: np.sum(x, axis=0))
events['EntitiesEmbS'] = events['EntitiesEmb'].apply(lambda x: np.sum(x, axis=0))
events['AuthorsEmbS'] = events['AuthorsEmb'].apply(lambda x: np.sum(x, axis=0))
events['allEmb'] = events['SubjectsEmbS'] + events['EntitiesEmbS'] + events['AuthorsEmbS'] +events['PaperEmb']

events['SubjectsEmbA'] = events['SubjectsEmb'].apply(lambda x: np.mean(x, axis=0))
events['EntitiesEmbA'] = events['EntitiesEmb'].apply(lambda x: np.mean(x, axis=0))
events['AuthorsEmbA'] = events['AuthorsEmb'].apply(lambda x: np.mean(x, axis=0))
events['meanEmb'] = events['SubjectsEmbA'] + events['EntitiesEmbA'] + events['AuthorsEmbA'] +events['PaperEmb']


allembs = events['allEmb'].tolist()
meanembs = events['meanEmb'].tolist()


matrixSum = np.array(allembs, dtype=np.float32)
matrixSumN = C_Optimizer.MinMaxNormalization(matrixSum)
matrixSumStd = sklearn.preprocessing.StandardScaler().fit_transform(matrixSum)

matrixMean = np.array(meanembs, dtype=np.float32)
matrixMeanN = C_Optimizer.MinMaxNormalization(matrixMean)
matrixMeanStd = sklearn.preprocessing.StandardScaler().fit_transform(matrixMean)


nbrs = NearestNeighbors(n_neighbors=5,algorithm='brute',metric = 'cosine').fit(matrixSumN)
distances, _ = nbrs.kneighbors(matrixSumN)
diff = np.apply_along_axis(np.diff, 1, distances)
diffSumN = distances[range(len(events)), np.apply_along_axis(np.argmax, 1, diff) + 1]
# diffSumN = np.apply_along_axis(np.mean, 1, distances)

nbrs = NearestNeighbors(n_neighbors=5,algorithm='brute',metric = 'cosine').fit(matrixSumStd)
distances, _ = nbrs.kneighbors(matrixSumStd)
diff = np.apply_along_axis(np.diff, 1, distances)
diffSumStd = distances[range(len(events)), np.apply_along_axis(np.argmax, 1, diff) + 1]
# diffSumStd = np.apply_along_axis(np.mean, 1, distances)

nbrs = NearestNeighbors(n_neighbors=5,algorithm='brute',metric = 'cosine').fit(matrixMeanN)
distances, _ = nbrs.kneighbors(matrixMeanN)
diff = np.apply_along_axis(np.diff, 1, distances)
diffMeanN = distances[range(len(events)), np.apply_along_axis(np.argmax, 1, diff) + 1]
# diffMeanN = np.apply_along_axis(np.mean, 1, distances)

nbrs = NearestNeighbors(n_neighbors=5,algorithm='brute',metric = 'cosine').fit(matrixMeanStd)
distances, _ = nbrs.kneighbors(matrixMeanStd)
diff = np.apply_along_axis(np.diff, 1, distances)
diffMeanStd = distances[range(len(events)), np.apply_along_axis(np.argmax, 1, diff) + 1]
# diffMeanStd = np.apply_along_axis(np.mean, 1, distances)



scored_events = events[['Title']].copy()
scored_events['ScoreSumN'] = diffSumN
scored_events['ScoreSumN'] = scored_events['ScoreSumN'].rank()
scored_events['ScoreSumStd'] = diffSumStd
scored_events['ScoreSumStd'] = scored_events['ScoreSumStd'].rank()
scored_events['ScoreMeanN'] = diffMeanN
scored_events['ScoreMeanN'] = scored_events['ScoreMeanN'].rank()
scored_events['ScoreMeanStd'] = diffMeanStd
scored_events['ScoreMeanStd'] = scored_events['ScoreMeanStd'].rank()

scored_events = scored_events.sort_values('ScoreSumN', ascending=False).reset_index(drop = True)

scored_events[['ScoreSumN','ScoreSumStd','ScoreMeanN','ScoreMeanStd']].corr()


# anomalies, info = parallelSTRAY(matrixSum, 10, alpha=0.15, threshold=10, proportion = 0.4); info
# scores = pd.DataFrame(info[1]['out_scores']); scores.describe()

# scored_events = events[['Title']].copy()
# scored_events['Score'] = scores
# scored_events['Distances'] = diff

# scored_events = scored_events.sort_values('Score', ascending=False)
# scored_events.head(100) 

# scored_events[scored_events['Title'].str.contains('MINE')]
# scored_events.describe()


# scored_events[scored_events['Title'].str.contains('The Usefulness of Multilevel Hash Tables ')]


## DISTINCT INTERESTS DECLARED BY AUTHRO IN A PAPER

paper_int = driver.execute_query("""
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
with distinct p.title as Title, a.interests as interests
RETURN Title, size(COLLECT(interests)) as Interests
""")
paper_int = pd.DataFrame(paper_int[0], columns = paper_int[-1])

anomalies = scored_events.iloc[15:500].copy()
non_anomalies = scored_events.iloc[500:].copy()

anomalies = pd.merge(anomalies, paper_int, on='Title', how='inner')
non_anomalies = pd.merge(non_anomalies, paper_int, on='Title', how='inner')

anomalies['Interests'].describe()
non_anomalies['Interests'].describe()

import scipy
scipy.stats.ks_2samp(anomalies['Interests'], non_anomalies['Interests'], alternative='two-sided', mode='auto')

# Plot histograms with density=True
plt.figure(figsize=(10, 6))

plt.hist(anomalies['Interests'], bins=20, density=True, alpha=0.5, label='Anomalies')
plt.hist(non_anomalies['Interests'], bins=20, density=True, alpha=0.5, label='Non-Anomalies')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Density')
# plt.title('Histograms of Anomalies and Non-Anomalies')
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# DISTINCT ENTITIES INVOLVED
entities = driver.execute_query("""
MATCH (e:Entity)<--(a:Author)-[:AUTHORED]->(p:Paper)
with distinct p.title as Title, e.name as nameentity
RETURN Title, size(COLLECT(nameentity)) as nameentity
""")

entities = pd.DataFrame(entities[0], columns = entities[-1])


anomalies = scored_events.iloc[15:500].copy()
non_anomalies = scored_events.iloc[500:].copy()

anomalies = pd.merge(anomalies, entities, on='Title', how='inner')

non_anomalies = pd.merge(non_anomalies, entities, on='Title', how='inner')

anomalies['nameentity'].describe()
non_anomalies['nameentity'].describe()


scipy.stats.ks_2samp(anomalies['nameentity'], non_anomalies['nameentity'], alternative='two-sided', mode='auto')

plt.figure(figsize=(10, 6))

# Plot histograms with density=True
plt.hist(anomalies['nameentity'], bins=20, density=True, alpha=0.5, label='Anomalies')
plt.hist(non_anomalies['nameentity'], bins=20, density=True, alpha=0.5, label='Non-Anomalies')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Density')
# plt.title('Histograms of Anomalies and Non-Anomalies')
plt.legend()
plt.grid(alpha=0.3)

plt.show()



# number of authors 
name = driver.execute_query("""
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
with distinct p.title as Title, a.name as name
RETURN Title, size(COLLECT(name)) as name
""")

name = pd.DataFrame(name[0], columns = name[-1])

anomalies = scored_events.iloc[15:500].copy()
non_anomalies = scored_events.iloc[500:].copy()

anomalies = pd.merge(anomalies, name, on='Title', how='inner')

non_anomalies = pd.merge(non_anomalies, name, on='Title', how='inner')

anomalies['name'].describe()
non_anomalies['name'].describe()

scipy.stats.ks_2samp(anomalies['name'], non_anomalies['name'], alternative='two-sided', mode='auto')

plt.figure(figsize=(10, 6))

# Plot histograms with density=True
plt.hist(anomalies['name'], bins=20, density=True, alpha=0.5, label='Anomalies')
plt.hist(non_anomalies['name'], bins=20, density=True, alpha=0.5, label='Non-Anomalies')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Density')
# plt.title('Histograms of Anomalies and Non-Anomalies')
plt.legend()
plt.grid(alpha=0.3)

plt.show()

