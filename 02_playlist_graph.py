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
import scipy

URI = "bolt://localhost:57190"
AUTH = ("", "")
driver = GraphDatabase.driver(URI, auth=AUTH)

# Generic statistics
numbergenres = driver.execute_query("""
MATCH (g:Genre)
RETURN COUNT(DISTINCT g) as count
""")

numberartist = driver.execute_query("""
MATCH (g:Artist)
RETURN COUNT(DISTINCT g) as count
""")

numbersong = driver.execute_query("""
MATCH (g:Song)
RETURN COUNT(DISTINCT g) as count
""")

avgArtistPerSong = driver.execute_query("""
MATCH (g:Song)<--(a:Artist)
RETURN g.title as title, COUNT(DISTINCT a) as count
""")

avgArtistPerSong = pd.DataFrame(avgArtistPerSong[0], columns = avgArtistPerSong[-1])
avgArtistPerSong.describe()

avgArtistPerSong = avgArtistPerSong[avgArtistPerSong['count']<15].reset_index(drop=True)
# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(avgArtistPerSong['count'], bins=20, alpha=0.5, label='Artist per Song')
# Add labels, legend, and title
plt.xlabel('Number of Artist per Song')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)



avgSongPerPlaylist = driver.execute_query("""
MATCH (g:Playlist)<--(a:Song)
RETURN g.id, COUNT(DISTINCT a) as count
""")

avgSongPerPlaylist = pd.DataFrame(avgSongPerPlaylist[0], columns = avgSongPerPlaylist[-1])
avgSongPerPlaylist.describe()
# avgSongPerPlaylist = avgSongPerPlaylist[avgSongPerPlaylist['count']<15].reset_index(drop=True)

# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(avgSongPerPlaylist['count'], bins=20, alpha=0.5, density = 'True')
# Add labels, legend, and title
plt.xlabel('Number of Song per Playlist')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)


avgGenrePerArtist = driver.execute_query("""
MATCH (a:Artist)-->(g:Genre)
RETURN a.name, COUNT(DISTINCT g) as count
""")

avgGenrePerArtist = pd.DataFrame(avgGenrePerArtist[0], columns = avgGenrePerArtist[-1])
avgGenrePerArtist.describe()
avgGenrePerArtist = avgGenrePerArtist[avgGenrePerArtist['count']<15].reset_index(drop=True)
# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(avgGenrePerArtist['count'], bins=12, alpha=0.5, density = 'True')
# Add labels, legend, and title
plt.xlabel('Number of Genres per Artist')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)



df = pd.read_csv('./data/playlist/playlist.csv', index_col = 0)
df = df.ID.tolist()
len(df)

# ANOMALIES ANALYSIS
anomalies = driver.execute_query("""
MATCH (p:Playlist)<--(s:Song)<--(a:Artist)-->(g:Genre)
WHERE p.id IN $anom
RETURN p.id AS ID, g.id as Genre
""", parameters_ = {'anom':df})

stat_genre = pd.DataFrame(anomalies[0], columns = anomalies[-1])

notanomalies = driver.execute_query("""
MATCH (p:Playlist)<--(s:Song)<--(a:Artist)-->(g:Genre)
WHERE NOT p.id IN $anom
RETURN p.id AS ID, g.id as Genre
""", parameters_ = {'anom':df})

stat_genre_not = pd.DataFrame(notanomalies[0], columns = notanomalies[-1])


# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(stat_genre.groupby('ID').size().reset_index()[0], bins=30, alpha=0.5, label='Anomalies', density=True)
plt.hist(stat_genre_not.groupby('ID').size().reset_index()[0], bins=30, alpha=0.5, label='Non-anomalies', density=True)
# Add labels, legend, and title
plt.xlabel('Number of Distinct Genres')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)



# ANOMALIES ANALYSIS
anomalies = driver.execute_query("""
MATCH (p:Playlist)<--(s:Song)<--(a:Artist)
WHERE p.id IN $anom
RETURN DISTINCT p.id AS ID, a.name as name
""", parameters_ = {'anom':df})

stat_artist = pd.DataFrame(anomalies[0], columns = anomalies[-1])

notanomalies = driver.execute_query("""
MATCH (p:Playlist)<--(s:Song)<--(a:Artist)
WHERE NOT p.id IN $anom
RETURN DISTINCT p.id AS ID, a.name as name
""", parameters_ = {'anom':df})

stat_artist_not = pd.DataFrame(notanomalies[0], columns = notanomalies[-1])


# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(stat_artist.groupby('ID').size().reset_index()[0], bins=30, alpha=0.5, label='Anomalies', density=True)
plt.hist(stat_artist_not.groupby('ID').size().reset_index()[0], bins=30, alpha=0.5, label='Non-anomalies', density=True)
# Add labels, legend, and title
plt.xlabel('Number of Distinct Artist')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)


# ANOMALIES ANALYSIS
anomalies = driver.execute_query("""
MATCH (p:Playlist)<--(s:Song)
WHERE p.id IN $anom
RETURN DISTINCT p.id AS ID, s.title AS title, s.releaseDate as date
""", parameters_ = {'anom':df})

stat_song = pd.DataFrame(anomalies[0], columns = anomalies[-1])

notanomalies = driver.execute_query("""
MATCH (p:Playlist)<--(s:Song)
WHERE NOT p.id IN $anom
RETURN DISTINCT p.id AS ID, s.title AS title, s.releaseDate as date
""", parameters_ = {'anom':df})

stat_song_not = pd.DataFrame(notanomalies[0], columns = notanomalies[-1])


# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(stat_song.groupby('ID').size().reset_index()[0], bins=5, alpha=0.5, label='Anomalies', density=True)
plt.hist(stat_song_not.groupby('ID').size().reset_index()[0], bins=5, alpha=0.5, label='Non-anomalies', density=True)
# Add labels, legend, and title
plt.xlabel('Number of Distinct Song Title')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)


stat_song['date'] = pd.to_datetime(stat_song['date'],format='mixed')
stat_song['year'] = stat_song['date'].apply(lambda x: x.year)

stat_song_not['year'] = stat_song_not['date'].apply(lambda x: x[:4])


# Set up the figure
plt.figure(figsize=(10, 6))
# Histogram
plt.hist(stat_song.groupby('ID')['year'].agg(np.median).reset_index()['year'], bins=20, alpha=0.5, label='Anomalies', density=True)
plt.hist(stat_song_not.groupby('ID')['year'].agg(np.median).reset_index()['year'], bins=20, alpha=0.5, label='Non-anomalies', density=True)
# Add labels, legend, and title
plt.xlabel('Number of Distinct Song Title')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)

x = stat_song.groupby('ID')['year'].agg(np.median).reset_index()['year']
y = stat_song_not.groupby('ID')['year'].agg(np.median).reset_index()['year']
scipy.stats.kstest( x,y, alternative='two-sided')
