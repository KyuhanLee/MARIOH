import itertools
import numpy as np

def extract_features(data, G, Gtrain_cliques_set):
    neighbors_dict = {node: set(G.neighbors(node)) for node in G.nodes()}
    neighbor_weights = {
        node: {nbr: G[node][nbr]['weight'] for nbr in G.neighbors(node)} for node in G.nodes()
    }
    edges_set = set(G.edges())

    result = []
    for clique in data:
        size = len(clique)
        edges = list(itertools.combinations(clique, 2))
        existing_edges = [
            (u, v) for u, v in edges if (u, v) in edges_set or (v, u) in edges_set
        ]
        weights = [
            G[u][v]['weight'] if (u, v) in edges_set else G[v][u]['weight']
            for u, v in existing_edges
        ]

        total_weight_internal = sum(weights)

        degrees = [G.degree(n, weight='weight') for n in clique]
        sum_degree = sum(degrees)
        avg_degree = np.mean(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        std_degree = np.std(degrees) if degrees else 0

        total_weight_va_to_v = sum_degree
        total_weight_va_to_non_va = total_weight_va_to_v - 2 * total_weight_internal

        if total_weight_va_to_v == 0:
            edge_cut_ratio = 1
        else:
            edge_cut_ratio = 1 - (total_weight_va_to_non_va / total_weight_va_to_v)

        total_weight = sum(weights)
        avg_weight = np.mean(weights) if weights else 0
        min_weight = min(weights) if weights else 0
        max_weight = max(weights) if weights else 0
        weight_std = np.std(weights) if weights else 0

        involved_first = 1 if tuple(sorted(clique)) in Gtrain_cliques_set else 0

        partial_ws, partial_amb = [], []
        for u, v in existing_edges:
            w = G[u][v]['weight'] if (u, v) in edges_set else G[v][u]['weight']
            common_neighbors = neighbors_dict[u] & neighbors_dict[v]
            weight_sum = sum(
                min(neighbor_weights[u][z], neighbor_weights[v][z]) for z in common_neighbors
            )
            partial_ws.append(weight_sum)
            partial_amb.append(weight_sum / w if w != 0 else 0)

        if partial_ws:
            partial_features = [
                sum(partial_ws), min(partial_ws), np.mean(partial_ws), max(partial_ws), np.std(partial_ws)
            ]
            partial_amb_features = [
                sum(partial_amb), min(partial_amb), np.mean(partial_amb), max(partial_amb), np.std(partial_amb)
            ]
        else:
            partial_features = [0, 0, 0, 0, 0]
            partial_amb_features = [0, 0, 0, 0, 0]

        result.append([
            size, total_weight, min_weight, avg_weight, max_weight, weight_std,
            sum_degree, min_degree, avg_degree, max_degree, std_degree,
            *partial_features, *partial_amb_features, involved_first, edge_cut_ratio
        ])
    return np.array(result)
