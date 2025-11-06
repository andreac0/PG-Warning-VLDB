import pandas as pd
import numpy as np
from aeon.anomaly_detection import STRAY
from faker import Faker
import random
from functions.GenerateKG import *
from functions.AnomaliesDetection import *
from functions.TextEmbedding import *
from datetime import timedelta
from datetime import datetime
import math
from random import sample
from scipy.spatial import distance

# Load C scaler for optimization
C_Optimizer = CFunctions()

# KG Generation
N = 50
R = 250
cnt = 'de_DE'
KG_IT_people = friendshipKG(country = cnt)


# KG Generation
nodes = KG_IT_people.get_nodes(N)

# Get Edge List --> progressive events
event_cards = []
date_str = '09-08-2022'
date_object = datetime.strptime(date_str, '%d-%m-%Y').date()

for i in range(R):
    event_cards.append(['Source: ' + parse_entity(nodes.sample(1), exclude = ['date','nationality']), 
                    f'Relationship: CONTACTS',
                    'Destination: ' +parse_entity(nodes.sample(1), exclude = ['date','nationality'])])
    date_object = date_object + timedelta(days=random.randint(1,2))



### Compute Embedding of Events
# Multilingual E5 Model
e5model = E5EmbeddingCUDA()
embE5 = e5model.compute(flatten(event_cards), N = 15)
embE5 = recompose(event_cards, embE5)

# BGE Model
bgeModel = BGEEmbeddingCUDA()
embBGE = bgeModel.compute(flatten(event_cards), N = 20)
embBGE = recompose(event_cards, embBGE)


# SET STRAY
alpha = 0.05
threshold = 20
proportion = 0.85

#----- with E5 Embeddings
anomalies, info = computeAnomalySTRAY(C_Optimizer, embE5, start_index = 2, end_index = len(embE5), 
                                alpha = alpha, proportion = proportion, threshold = threshold)
plot_anomalies(anomalies)
plot_bound(info)

#----- with BGE Embeddings
anomaliesBGE, infoBGE = computeAnomalySTRAY(C_Optimizer, embBGE, start_index = 2, end_index = len(embBGE), 
                                alpha = alpha, proportion = proportion, threshold = threshold)
plot_anomalies(anomaliesBGE)
plot_bound(infoBGE)


# xaxis = np.arange(0, len(anomalies))
# cumy = np.array(anomalies).cumsum()

# # Create a figure with custom size
# fig, axes = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)

# # Plot step object
# axes.step(xaxis, cumy, where='mid', label='Warnings',  linewidth=5)
# axes.set_xlabel("Subsequence of Synthetic Events", fontsize=12)
# axes.set_ylabel("Cumulative Number of Warnings", fontsize=12)
# # axes.legend()
# axes.grid(True)


# # Show the plots
# plt.show()


detector = STRAY(alpha = alpha, k=int(math.sqrt(1024)), size_threshold = threshold, p = proportion)

# Add anomaly as R+1 event
KG_Foreign_people = friendshipKG(country = 'de_DE')

is_anomaly = []
for i in range(100):
    nodes_fr = KG_Foreign_people.get_nodes(5)
    # -- Event
    event = ['Source: ' + parse_entity(nodes_fr.sample(1), exclude = ['date','nationality']),
            f'Relationship: CONTACTS', 
            'Destination: ' + parse_entity(nodes.sample(1), exclude = ['date','nationality'])]
    # -- compute embedding
    newembE5 = embE5 + [[sum(x) for x in zip(*e5model.compute(event))]]
    y = detector.fit_predict(C_Optimizer.MinMaxNormalization(np.array(newembE5, dtype=np.float32)), axis=0)
    is_anomaly.append(y[0][-1])

# source
print(sum(is_anomaly))


is_anomaly = []
for i in range(100):
    nodes_fr = KG_Foreign_people.get_nodes(5)
    # -- Event
    event = ['Source: ' + parse_entity(nodes.sample(1), exclude = ['date','nationality']),
            f'Relationship: CONTACTS', 
            'Destination: ' + parse_entity(nodes_fr.sample(1), exclude = ['date','nationality'])]
    # -- compute embedding
    newembE5 = embE5 + [[sum(x) for x in zip(*bgeModel.compute(event))]]
    y = detector.fit_predict(C_Optimizer.MinMaxNormalization(np.array(newembE5, dtype=np.float32)), axis=0)
    is_anomaly.append(y[0][-1])

# dest
print(sum(is_anomaly))


# Add anomaly as R+1 event
is_anomaly = []
for i in range(100):
    nodes_fr = KG_Foreign_people.get_nodes(5)
    # -- Event
    event = ['Source: ' + parse_entity(nodes_fr.sample(1), exclude = ['date','nationality']),
            f'Relationship: CONTACTS', 
            'Destination: ' + parse_entity(nodes.sample(1), exclude = ['date','nationality'])]
    # -- compute embedding
    newembBGE = embBGE + [[sum(x) for x in zip(*bgeModel.compute(event))]]
    y = detector.fit_predict(C_Optimizer.MinMaxNormalization(np.array(newembBGE, dtype=np.float32)), axis=0)
    is_anomaly.append(y[0][-1])

# source
print(sum(is_anomaly))

is_anomaly = []
for i in range(100):
    nodes_fr = KG_Foreign_people.get_nodes(5)
    # -- Event
    event = ['Source: ' + parse_entity(nodes.sample(1), exclude = ['date','nationality']),
            f'Relationship: CONTACTS', 
            'Destination: ' + parse_entity(nodes_fr.sample(1), exclude = ['date','nationality'])]
    # -- compute embedding
    newembBGE = embBGE + [[sum(x) for x in zip(*e5model.compute(event))]]
    y = detector.fit_predict(C_Optimizer.MinMaxNormalization(np.array(newembBGE, dtype=np.float32)), axis=0)
    is_anomaly.append(y[0][-1])
