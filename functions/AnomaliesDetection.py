import ctypes
import numpy as np
from aeon.anomaly_detection import STRAY
import aeon
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class CFunctions:

    def __init__(self):
        # Load the shared library
        self.c_lib = ctypes.CDLL('functions/MinMax/minmax.so')

        # Define the function signature
        self.c_lib.min_max_normalize_columns.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Input matrix (float*)
            ctypes.c_size_t,                 # Number of rows (size_t)
            ctypes.c_size_t,                 # Number of columns (size_t)
            ctypes.c_float,                  # New min value (float)
            ctypes.c_float                   # New max value (float)
        ]
        # c_lib.min_max_normalize_columns.restype = None
        self.c_lib.min_max_normalize_columns.restype = ctypes.POINTER(ctypes.c_float)

    def MinMaxNormalization(self, matrix):

        # matrix = np.array(lists_of_embedding, dtype=np.float32)

        matrix_flat = matrix.flatten()

        # Create an output matrix (same size as the input matrix)
        # Call the C function
        rows, cols = matrix.shape
        # Cast pointers to writable memory
        matrix_ptr = ctypes.cast(matrix_flat.ctypes.data, ctypes.POINTER(ctypes.c_float))

        normalized_matrix_flat = self.c_lib.min_max_normalize_columns(
            matrix_ptr,
            rows,
            cols,
            ctypes.c_float(0.0),
            ctypes.c_float(1.0)
        )
        normalized_matrix_flat = np.ctypeslib.as_array(normalized_matrix_flat, shape=(rows,cols))

        return normalized_matrix_flat
    
    def MinMaxColNormalization(self, column):

        column = column.flatten()
        # Create an output matrix (same size as the input matrix)
        # Call the C function
        rows, cols = len(column), 1
        # Cast pointers to writable memory
        column_ptr = ctypes.cast(column.ctypes.data, ctypes.POINTER(ctypes.c_float))

        normalized_column = self.c_lib.min_max_normalize_columns(
            column_ptr,
            rows,
            cols,
            ctypes.c_float(0.0),
            ctypes.c_float(1.0)
        )
        normalized_column = np.ctypeslib.as_array(normalized_column, shape=(rows,cols))

        return normalized_column
    
def parallelSTRAY(prepD, i, alpha=0.15, threshold=20, proportion = 0.5):
    detector = STRAY(alpha = alpha, k=int(i), size_threshold = threshold, p = proportion)
    y = detector.fit_predict(np.array(prepD), axis=0)
    prev_anomalies = y[0][-1]
    
    return prev_anomalies, y


def flatten(xss):
    return [x for xs in xss for x in xs]

def recompose(event_cards, emb_event):
    cardEmb = []
    n = 0
    j = 1
    for j in range(len(event_cards)):
        N = len(event_cards[j])
        cardEmb = cardEmb + [[sum(x) for x in zip(*emb_event[n:(N+n)])]]
        n = n+N
    
    return cardEmb

def recomposePYARROW(event_cards, emb_event):
    cardEmb = []
    n = 0
    j = 1
    for j in range(len(event_cards)):
        N = len(event_cards[j])
        cardEmb = cardEmb + [[sum(scalar.as_py() for scalar in x) for x in zip(*emb_event[n:(N+n)])]]
        n = n+N
    
    return cardEmb


def plot_anomalies(anomalies):
    
    xaxis = np.arange(0, len(anomalies))
    yaxis = np.array(anomalies)
    cumy = np.array(anomalies).cumsum()

    # Create a figure with custom size
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Plot step object
    axes[0].step(xaxis, yaxis, where='mid', label='Warnings',  linewidth=0.5)
    axes[0].set_xlabel("Event", fontsize=12)
    axes[0].set_ylabel("Warning", fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # Plot cumulative sum
    axes[1].plot(xaxis, cumy, linestyle='-', label='Cumulative Sum', color='g', linewidth=1)
    axes[1].set_xlabel("Event", fontsize=12)
    axes[1].set_ylabel("Cumulative Sum", fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    # Show the plots
    plt.show()

def plot_bound(info, start = 0):

    bounds = []
    for j in range(start, len(info)):
        bounds.append(info[j][1]['bound'])

    plt.plot(bounds)


def computeAnomalySTRAY(scalerC, embedded_cards, start_index, end_index, alpha = 0.15, proportion = 0.5, threshold = 20, M = 150, decay = 10, minM = 2):
    
    # NORMALIZATION + STRAY
    prev_anomalies, all_info = [], []

    # SET INDEXES OF EVENTS
    matrix = np.array(embedded_cards[:end_index], dtype=np.float32)
    global_min = np.minimum(0, np.min(embedded_cards[:start_index], axis=0))
    global_max = np.maximum(0, np.max(embedded_cards[:start_index], axis=0))
    prev_min = global_min.copy()
    prev_max = global_max.copy()

    preprocessedData = scalerC.MinMaxNormalization(matrix[:start_index])

    strayData = []
    strayData.append(preprocessedData)

    for i in tqdm(range(start_index,end_index)):

        if i % M == 0:
            m = [1024]*M#[*range(i-(M-1),i+1)]

            with ThreadPoolExecutor(80) as executor:
                results = list(executor.map(parallelSTRAY, strayData, m, [alpha]*M, [threshold]*M, [proportion]*M))

            for j in range(len(results)):
                prev_anomalies.append(results[j][0])
                all_info.append(results[j][1])

            strayData = []

            if M < minM:
                M = minM
            elif M > minM:
                M = M-decay
                
        global_min = np.minimum(global_min, embedded_cards[i])
        global_max = np.maximum(global_max, embedded_cards[i])

        if sum(global_min == prev_min) == 1024 and sum(global_max == prev_max) == 1024:
            detect_anomalies = np.vstack([preprocessedData,(embedded_cards[i]-global_min)/(global_max-global_min)])
        else:
            if i < 4000:
                detect_anomalies = scalerC.MinMaxNormalization(matrix[:(i+1)])
            else:
                with ThreadPoolExecutor(80) as executor:
                    results = list(executor.map(scalerC.MinMaxColNormalization, np.hsplit(matrix[:(i+1)], matrix.shape[1])))
                    detect_anomalies = np.hstack(results)
            prev_min = global_min.copy()
            prev_max = global_max.copy()

        strayData.append(detect_anomalies)
        preprocessedData = detect_anomalies.copy()

        if M == 0:
            M = minM

    m = [1024]*len(strayData)

    with ThreadPoolExecutor(80) as executor:
        results = list(executor.map(parallelSTRAY, strayData, m, [alpha]*M, [threshold]*M, [proportion]*M))
    for j in range(len(results)):
        prev_anomalies.append(results[j][0])
        all_info.append(results[j][1])
    
    return prev_anomalies, all_info


def getMostSimilarEvent(all_events, embeddings, index_event_of_interest, neighbors = 35):

    nbrs = NearestNeighbors(
        n_neighbors=neighbors,
        algorithm='brute',
        metric = 'cosine').fit(embeddings)
    distances, close_events = nbrs.kneighbors(embeddings)

    close_events = close_events[index_event_of_interest][1:]

    return all_events[index_event_of_interest], np.array(all_events)[list(close_events)], close_events
