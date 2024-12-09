from collections import Counter
import itertools
import networkx as nx

def evaluate_graph(G, gt, pred):
    jaccard_similarity = jaccard_similarity_hyperedges(pred, gt)
    multiset_jaccard_similarity_value = multiset_jaccard_similarity(pred, gt)
    duplicity = 100 * (1 - (len(set(pred)) / len(pred))) if len(pred) > 0 else 0
    sanity_check_result = sanity_check(G, pred)

    print("Jaccard Similarity:", jaccard_similarity)
    print("Multiset Jaccard Similarity:", multiset_jaccard_similarity_value)
    print("Duplicity:", duplicity)
    print("Sanity Check Passed:", sanity_check_result)

    return jaccard_similarity, multiset_jaccard_similarity_value, duplicity, sanity_check_result

def jaccard_similarity_hyperedges(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def multiset_jaccard_similarity(set1, set2):
    count_set1 = Counter(set1)
    count_set2 = Counter(set2)
    keys = count_set1.keys() | count_set2.keys()
    intersection_sum = sum(min(count_set1[k], count_set2[k]) for k in keys)
    union_sum = sum(max(count_set1[k], count_set2[k]) for k in keys)
    return intersection_sum / union_sum if union_sum != 0 else 0

def sanity_check(G, pred_hyperedges):
    G_pred = nx.Graph()
    for hyperedge in pred_hyperedges:
        for u, v in itertools.combinations(hyperedge, 2):
            if G_pred.has_edge(u, v):
                G_pred[u][v]['weight'] += 1
            else:
                G_pred.add_edge(u, v, weight=1)

    for u, v, data in G.edges(data=True):
        if not G_pred.has_edge(u, v) or G_pred[u][v]['weight'] != data['weight']:
            return False

    for u, v, data in G_pred.edges(data=True):
        if not G.has_edge(u, v):
            return False

    return True
