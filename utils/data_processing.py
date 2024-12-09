import os
import random
import numpy as np
import torch
from collections import defaultdict
import itertools
import networkx as nx
from joblib import Parallel, delayed

from utils.graph_operations import (
    select_edges_by_weight_condition_reduced,
    select_edges_by_weight_condition_preserved
)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def preprocess_data(data_path):
    edge_weights = defaultdict(int)
    gt = set()
    with open(data_path, "r") as f:
        for line in f:
            nodes = list(map(int, line.strip().split()))
            if len(nodes) <= 1:
                continue
            sorted_nodes = tuple(sorted(nodes))
            gt.add(sorted_nodes)
            for u, v in itertools.combinations(nodes, 2):
                edge = (min(u, v), max(u, v))
                edge_weights[edge] += 1

    G = nx.Graph()
    G.add_edges_from(((u, v, {'weight': w}) for (u, v), w in edge_weights.items()))
    return G, gt

def preprocess_data_preserved(data_path):
    edge_weights = defaultdict(int)
    gt = []
    with open(data_path, "r") as f:
        for line in f:
            nodes = list(map(int, line.strip().split()))
            if len(nodes) <= 1:
                continue
            sorted_nodes = tuple(sorted(nodes))
            gt.append(sorted_nodes)
            for u, v in itertools.combinations(nodes, 2):
                edge = (min(u, v), max(u, v))
                edge_weights[edge] += 1

    G = nx.Graph()
    G.add_edges_from(((u, v, {'weight': w}) for (u, v), w in edge_weights.items()))
    return G, gt

def load_and_preprocess_data(data_, preserved=False):
    if preserved:
        train_path = f"data/{data_}/train_dup.txt"
        test_path = f"data/{data_}/test_dup.txt"
        results = Parallel(n_jobs=2)(delayed(preprocess_data_preserved)(path) for path in [train_path, test_path])
        G, gt = results[0]
        G_test, gt_test = results[1]

        m_gt, m_G = select_edges_by_weight_condition_preserved(gt, G.copy())
        m_gt_test, m_G_test = select_edges_by_weight_condition_preserved(gt_test, G_test.copy())
    else:
        train_path = f"data/{data_}/train.txt"
        test_path = f"data/{data_}/test.txt"
        results = Parallel(n_jobs=2)(delayed(preprocess_data)(path) for path in [train_path, test_path])
        G, gt = results[0]
        G_test, gt_test = results[1]

        m_gt, m_G = select_edges_by_weight_condition_reduced(gt, G.copy())
        m_gt_test, m_G_test = select_edges_by_weight_condition_reduced(gt_test, G_test.copy())

    Gtrain_cliques = {tuple(sorted(clique)) for clique in nx.find_cliques(m_G)}
    Gtest_cliques = {tuple(sorted(clique)) for clique in nx.find_cliques(m_G_test)}

    return G, gt, G_test, gt_test, Gtrain_cliques, Gtest_cliques, m_gt, m_gt_test, m_G, m_G_test

def store_cliques_by_size(maximal_cliques):
    from collections import defaultdict
    cliques_by_size = defaultdict(list)
    for clique in maximal_cliques:
        cliques_by_size[len(clique)].append(clique)
    return cliques_by_size

def generate_subclique(clique, target_size):
    import random
    return tuple(sorted(random.sample(clique, target_size)))

def prepare_sns_data(gt, K, Gtrain_cliques):
    gt_set = set(gt)
    cliques_by_size = store_cliques_by_size(Gtrain_cliques)
    false_cliques_sampled = set()

    for clique in gt:
        size = len(clique)
        larger_cliques = [c for s, cliques in cliques_by_size.items() for c in cliques if s >= size]
        attempts = 0
        max_attempts = K * 5
        while len(false_cliques_sampled) < len(gt_set) * K and attempts < max_attempts:
            if larger_cliques:
                source_clique = random.choice(larger_cliques)
                new_clique = generate_subclique(source_clique, size)
                if new_clique not in gt_set and new_clique not in false_cliques_sampled:
                    false_cliques_sampled.add(new_clique)
            attempts += 1

    false_cliques_list = list(false_cliques_sampled)
    if len(false_cliques_list) > len(gt_set) * K:
        false_cliques_list = random.sample(false_cliques_list, len(gt_set) * K)

    X = list(gt_set) + false_cliques_list
    labels = [1] * len(gt_set) + [0] * len(false_cliques_list)
    return X, labels

def normalize_features(true_features, false_features):
    combined_features = np.vstack((true_features, false_features))
    epsilon = 1e-5
    X_mean = combined_features.mean(axis=0)
    X_std = combined_features.std(axis=0) + epsilon
    normalized_true = (true_features - X_mean) / X_std
    normalized_false = (false_features - X_mean) / X_std
    return normalized_true, normalized_false, X_mean, X_std
