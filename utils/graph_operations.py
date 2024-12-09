from collections import defaultdict
import itertools
import networkx as nx
import random
import math
import torch
import torch.nn.functional as F

def select_edges_by_weight_condition_reduced(gt, graph):
    gt_set = gt.copy()
    edge_decrease = defaultdict(int)

    neighbor_weights = {node: {nbr: data['weight'] for nbr, data in graph[node].items()} for node in graph.nodes()}

    for u, v, w in graph.edges(data='weight'):
        neighbors_u = neighbor_weights[u]
        neighbors_v = neighbor_weights[v]
        common_neighbors = set(neighbors_u.keys()) & set(neighbors_v.keys())
        
        weight_sum = sum(min(neighbors_u[z], neighbors_v[z]) for z in common_neighbors)
        left_over = w - weight_sum

        if left_over > 0:
            edge = (min(u, v), max(u, v))
            edge_decrease[edge] += left_over

    for edge, decrease in edge_decrease.items():
        u, v = edge
        current_weight = graph[u][v]['weight']
        new_weight = current_weight - decrease
        if new_weight <= 0:
            graph.remove_edge(u, v)
        else:
            graph[u][v]['weight'] = new_weight

        if edge in gt_set:
            gt_set.discard(edge)

    return gt_set, graph

def select_edges_by_weight_condition_preserved(gt, graph):
    gt_set = gt.copy() if isinstance(gt, set) else set(gt)
    edge_decrease = defaultdict(int)

    neighbor_weights = {node: {nbr: data['weight'] for nbr, data in graph[node].items()} for node in graph.nodes()}

    for u, v, w in graph.edges(data='weight'):
        neighbors_u = neighbor_weights[u]
        neighbors_v = neighbor_weights[v]
        common_neighbors = set(neighbors_u.keys()) & set(neighbors_v.keys())
        
        weight_sum = sum(min(neighbors_u[z], neighbors_v[z]) for z in common_neighbors)
        left_over = w - weight_sum

        if left_over > 0:
            edge = (min(u, v), max(u, v))
            edge_decrease[edge] += left_over
            gt_set = [e for e in gt_set if e != edge or (left_over := left_over - 1) < 0]

    for edge, decrease in edge_decrease.items():
        u, v = edge
        current_weight = graph[u][v]['weight']
        new_weight = current_weight - decrease
        if new_weight <= 0:
            graph.remove_edge(u, v)
        else:
            graph[u][v]['weight'] = new_weight

    return gt_set, graph

def generate_data(G, cliques, mode, X_mean, X_std, Gtrain_cliques, Gtest_cliques):
    from utils.feature_extraction import extract_features
    if mode == 'train':
        features = extract_features(cliques, G, Gtrain_cliques)
    else:
        features = extract_features(cliques, G, Gtest_cliques)
    ans = (features - X_mean) / X_std
    return ans

def generate_pred_hyperedges(G, gt, classifier_net, X_mean, X_std, device, mode, best_thres=0.8, ratio_prob=20, Gtrain_cliques=None, Gtest_cliques=None):
    def select_edges_by_weight_condition(graph, pred):
        edge_decrease = defaultdict(int)
        neighbor_weights = {node: {nbr: data['weight'] for nbr, data in graph[node].items()} for node in graph.nodes()}
    
        for u, v, w in graph.edges(data='weight'):
            neighbors_u = neighbor_weights[u]
            neighbors_v = neighbor_weights[v]
            common_neighbors = set(neighbors_u.keys()) & set(neighbors_v.keys())
            
            weight_sum = sum(min(neighbors_u[z], neighbors_v[z]) for z in common_neighbors)
            left_over = w - weight_sum
    
            if left_over > 0:
                edge = (min(u, v), max(u, v))
                edge_decrease[edge] += left_over
                pred.extend([edge] * left_over)
    
        for edge, decrease in edge_decrease.items():
            u, v = edge
            current_weight = graph[u][v]['weight']
            new_weight = current_weight - decrease
            if new_weight <= 0:
                graph.remove_edge(u, v)
            else:
                graph[u][v]['weight'] = new_weight
        return pred

    G_ = G.copy()
    G_tmp_train = G.copy()
    pred = []
    pred = select_edges_by_weight_condition(G_tmp_train, pred)

    phase_num = 0
    initial_temp = best_thres
    final_temp = 0
    num_iterations = 20
    decrement = (initial_temp - final_temp) / num_iterations
    temperature = initial_temp

    while G_tmp_train.edges:
        all_cliques = list(nx.find_cliques(G_tmp_train))
        filtered_cliques = [clique for clique in all_cliques if len(clique) > 1]
        candid_data = generate_data(G_, filtered_cliques, mode, X_mean, X_std, Gtrain_cliques, Gtest_cliques)
        candid_data = torch.tensor(candid_data, dtype=torch.float32).to(device)
        logits = classifier_net(candid_data)
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predicted_labels = (probabilities > temperature).int()
        true_indices = torch.where(predicted_labels == 1)[0]
        sorted_true_indices = true_indices[torch.argsort(probabilities[true_indices], descending=True)]
        edge_clique = defaultdict(list)

        for index in sorted_true_indices:
            tmp_c = filtered_cliques[index]
            for edge in itertools.combinations(tmp_c, 2):
                sedge = tuple(sorted(edge))
                edge_clique[sedge].append(index)

        for index in sorted_true_indices:
            X = filtered_cliques[index]
            if X is None:
                continue
            pr = tuple(sorted(map(int, X)))
            pred.append(pr)
            for nCr in itertools.combinations(X, 2):
                sE = sorted(nCr)
                if index in edge_clique[(sE[0], sE[1])]:
                    edge_clique[(sE[0], sE[1])].remove(index)
                tmp = G_tmp_train.edges[nCr]['weight']
                if tmp == 1:
                    G_tmp_train.remove_edge(nCr[0], nCr[1])
                    for clique_index in edge_clique[(sE[0], sE[1])]:
                        filtered_cliques[clique_index] = None
                else:
                    G_tmp_train.add_edge(nCr[0], nCr[1], weight=(tmp - 1))

        false_indices = torch.where(predicted_labels == 0)[0]
        num_indices = math.ceil(len(false_indices) * (ratio_prob / 100))
        lowest_x_percent_indices = false_indices[torch.argsort(probabilities[false_indices], descending=False)[:num_indices]]
        removableCan = set()
        for index in lowest_x_percent_indices:
            X = filtered_cliques[index]
            for pk in range(2, len(X)):
                combinations = random.sample(X, pk)
                if combinations:
                    tmpX = tuple(sorted(combinations))
                    removableCan.add(tmpX)

        if len(removableCan) == 0:
            phase_num += 1
            temperature -= decrement
            continue

        removableCan = [list(tup) for tup in removableCan]

        candid_data = generate_data(G_, removableCan, mode, X_mean, X_std, Gtrain_cliques, Gtest_cliques)
        candid_data = torch.tensor(candid_data, dtype=torch.float32).to(device)
        logits = classifier_net(candid_data)
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predicted_labels = (probabilities > temperature).int()
        true_indices = torch.where(predicted_labels == 1)[0]
        sorted_true_indices = true_indices[torch.argsort(probabilities[true_indices], descending=True)]
        edge_clique = defaultdict(list)
        for index in sorted_true_indices:
            tmp_c = removableCan[index]
            for edge in itertools.combinations(tmp_c, 2):
                sedge = tuple(sorted(edge))
                edge_clique[sedge].append(index)

        for index in sorted_true_indices:
            X = removableCan[index]
            if X is None:
                continue
            connected = True
            for ppp in itertools.combinations(X, 2):
                if not G_tmp_train.has_edge(ppp[0], ppp[1]):
                    connected = False
                    break
            if not connected:
                continue
            pr = tuple(sorted(map(int, X)))
            pred.append(pr)
            for nCr in itertools.combinations(X, 2):
                sE = sorted(nCr)
                if index in edge_clique[(sE[0], sE[1])]:
                    edge_clique[(sE[0], sE[1])].remove(index)
                tmp = G_tmp_train.edges[nCr]['weight']
                if tmp == 1:
                    G_tmp_train.remove_edge(nCr[0], nCr[1])
                    for clique_index in edge_clique[(sE[0], sE[1])]:
                        removableCan[clique_index] = None
                else:
                    G_tmp_train.add_edge(nCr[0], nCr[1], weight=(tmp - 1))

        phase_num += 1
        temperature -= decrement

    return pred