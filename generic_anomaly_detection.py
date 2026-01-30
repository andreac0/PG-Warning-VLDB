import pandas as pd
import numpy as np
from functions.AnomaliesDetection import CFunctions
from neo4j import GraphDatabase
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing

def detect_anomalies(uri, auth, query, id_column, embedding_columns):
    """
    A generic function to detect anomalies in a graph.

    :param uri: The URI for the Neo4j database.
    :param auth: The authentication tuple for the Neo4j database.
    :param query: The Cypher query to execute to simulate the event generation. In a real environment this would be a stream of events accurately stored
    :param id_column: The name of the column to use as the identifier for the events.
    :param embedding_columns: A list of column names that contain embeddings.
    :return: A pandas DataFrame with the anomaly scores for each event.
    """
    # Load C scaler for optimization
    c_optimizer = CFunctions()

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=auth)

    # Execute query
    events = driver.execute_query(query)
    events_df = pd.DataFrame(events[0], columns=events[-1])

    # Combine embeddings
    events_df['allEmb'] = events_df[embedding_columns].apply(
        lambda row: np.sum(np.vstack(row), axis=0), axis=1
    )
    events_df['meanEmb'] = events_df[embedding_columns].apply(
        lambda row: np.mean(np.vstack(row), axis=0), axis=1
    )

    allembs = events_df['allEmb'].tolist()
    meanembs = events_df['meanEmb'].tolist()

    matrix_sum = np.array(allembs, dtype=np.float32)
    matrix_sum_n = c_optimizer.MinMaxNormalization(matrix_sum)
    matrix_sum_std = sklearn.preprocessing.StandardScaler().fit_transform(matrix_sum)

    matrix_mean = np.array(meanembs, dtype=np.float32)
    matrix_mean_n = c_optimizer.MinMaxNormalization(matrix_mean)
    matrix_mean_std = sklearn.preprocessing.StandardScaler().fit_transform(matrix_mean)

    # Calculate anomaly scores using Nearest Neighbors
    def get_nn_scores(matrix):
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(matrix)
        distances, _ = nbrs.kneighbors(matrix)
        diff = np.apply_along_axis(np.diff, 1, distances)
        return distances[range(len(events_df)), np.apply_along_axis(np.argmax, 1, diff) + 1]

    diff_sum_n = get_nn_scores(matrix_sum_n)
    diff_sum_std = get_nn_scores(matrix_sum_std)
    diff_mean_n = get_nn_scores(matrix_mean_n)
    diff_mean_std = get_nn_scores(matrix_mean_std)

    # Create a new DataFrame with the scores
    scored_events = events_df[[id_column]].copy()
    scored_events['ScoreSumN'] = diff_sum_n
    scored_events['ScoreSumN'] = scored_events['ScoreSumN'].rank()
    scored_events['ScoreSumStd'] = diff_sum_std
    scored_events['ScoreSumStd'] = scored_events['ScoreSumStd'].rank()
    scored_events['ScoreMeanN'] = diff_mean_n
    scored_events['ScoreMeanN'] = scored_events['ScoreMeanN'].rank()
    scored_events['ScoreMeanStd'] = diff_mean_std
    scored_events['ScoreMeanStd'] = scored_events['ScoreMeanStd'].rank()

    return scored_events.sort_values('ScoreSumN', ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. PAPERS GRAPH EXAMPLE
    print("Running anomaly detection on the 'Papers' graph...")
    URI_PAPERS = "bolt://localhost:17180"
    AUTH_PAPERS = ("", "")
    # HP: we get the set of events by querying the subgraphs of papers and their authors, subjects, and entities
    QUERY_PAPERS = """
    MATCH k=(e:Entity)<-[:AFFILIATED_WITH]-(a:Author)-[:AUTHORED]->(p:Paper)-[:IS_CLASSIFIED_AS]->(s:Subject)
    RETURN DISTINCT p.title AS Title, p.embedding as PaperEmb, COLLECT(s.embedding) AS SubjectsEmb, COLLECT(e.embedding) AS EntitiesEmb, COLLECT(a.embedding) AS AuthorsEmb
    """
    ID_COLUMN_PAPERS = 'Title'
    EMBEDDING_COLUMNS_PAPERS = ['PaperEmb', 'SubjectsEmb', 'EntitiesEmb', 'AuthorsEmb']

    try:
        papers_anomalies = detect_anomalies(URI_PAPERS, AUTH_PAPERS, QUERY_PAPERS, ID_COLUMN_PAPERS, EMBEDDING_COLUMNS_PAPERS)
        print("Top 5 anomalies in the 'Papers' graph:")
        print(papers_anomalies.head())
    except Exception as e:
        print(f"Could not run anomaly detection on 'Papers' graph: {e}")
