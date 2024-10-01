import numpy as np
from nltk.corpus import brown
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils.preprocessing import lemma_replacement

def use_agglomerative(embeddings: np.ndarray, 
                      lemmatized_texts: list) -> dict:
    """
    Use agglomerative clustering algorithm
    """
    agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='ward')
    labels = agglomerative.fit_predict(embeddings)

    word_clusters = defaultdict(list)
    for word, label in zip(lemmatized_texts, labels):
        word_clusters[label].append(word)

    return word_clusters

def use_dbSCAN(embeddings: np.ndarray, 
               lemmatized_texts: list) -> dict:
    """
    Use DBSCAN clustering algorithm
    """
    dbscan = DBSCAN(eps=0.8, min_samples=1)
    labels = dbscan.fit_predict(embeddings)

    word_clusters = defaultdict(list)
    for word, label in zip(lemmatized_texts, labels):
        word_clusters[label].append(word)

    return word_clusters

def print_clusters(word_clusters: dict) -> None:
    """
    Print clusters
    """
    print("Words in each cluster:")
    for label, cluster_words in word_clusters.items():
        print(f"Cluster {label}: {', '.join(cluster_words)}")

def select_most_frequent_word(cluster_words: list) -> str:
    """
    Select most frequent word
    """
    brown_freq = Counter(brown.words())
    word_counts = Counter(cluster_words)
    sorted_words = sorted(word_counts.items(), 
                          key=lambda x: (-x[1], -brown_freq.get(x[0], 0)))
    return sorted_words[0][0]

def replace_keys_by_function(d, func):
    """
    Replace keys by function
    """
    return {func(k): v for k, v in d.items()}

def get_final_dict(word_clusters: dict) -> None:
    """
    Get final word dictionary
    """
    final_word_dict = {}
    for label, cluster_words in word_clusters.items():
        most_frequent_word = select_most_frequent_word(cluster_words)
        final_word_dict[most_frequent_word] = len(cluster_words)

    final_word_dict = replace_keys_by_function(final_word_dict, lemma_replacement)

    return final_word_dict