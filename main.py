import argparse
import time
import torch
import torch.nn as nn
import numpy as np

from params import params_dict_reduced, params_dict_preserved
from utils.data_processing import set_random_seed, load_and_preprocess_data, prepare_sns_data, normalize_features
from utils.feature_extraction import extract_features
from utils.training import CliqueDataset, train_classifier_net
from utils.graph_operations import generate_data, generate_pred_hyperedges
from utils.evaluation import evaluate_graph
from utils.io_utils import write_tuples_to_txt

def main():
    parser = argparse.ArgumentParser(description='Hyperedge Reconstruction')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--preserved', action='store_true', help='Use preserved mode (train_dup.txt etc.)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)

    start_time = time.time()
    G, gt, G_test, gt_test, Gtrain_cliques, Gtest_cliques, m_gt, m_gt_test, m_G, m_G_test = load_and_preprocess_data(args.data, preserved=args.preserved)
    print(f"Load done: {time.time() - start_time}")

    start_time = time.time()
    X, y = prepare_sns_data(m_gt, 4, Gtrain_cliques)
    print(f"Sampling done: {time.time() - start_time}")

    features = extract_features(X, m_G, Gtrain_cliques)
    true_features = features[(np.array(y) == 1)]
    false_features = features[(np.array(y) == 0)]

    true_features, false_features, X_mean, X_std = normalize_features(true_features, false_features)

    X_train = torch.tensor(
        (true_features.tolist() + false_features.tolist()), dtype=torch.float32
    )
    y_train = torch.tensor([1]*len(true_features) + [0]*len(false_features), dtype=torch.long)

    train_dataset = CliqueDataset(X_train, y_train)

    if args.preserved:
        params_dict = params_dict_preserved
    else:
        params_dict = params_dict_reduced

    best_thres = params_dict[args.data]['best_thres']
    ratio_prob = params_dict[args.data]['ratio_prob']
    best_params = params_dict[args.data]['best_params'].copy()
    del best_params['loss_type']

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    classifier_net = train_classifier_net(train_dataset, device, **best_params, criterion=criterion)
    print(f"Training done: {time.time() - start_time}")

    start_time = time.time()
    pred_hyperedges_test = generate_pred_hyperedges(G_test, gt_test, classifier_net, X_mean, X_std, device, 'test', best_thres, ratio_prob, Gtrain_cliques, Gtest_cliques)
    print(f"Execution Time: {time.time() - start_time}")

    evaluate_graph(G_test, gt_test, pred_hyperedges_test)

    mode_str = "preserved" if args.preserved else "reduced"
    file_path = f'{args.output_dir}/reconstructed_hyp_{mode_str}/{args.data}_{args.seed}.txt'
    write_tuples_to_txt(pred_hyperedges_test, file_path)

if __name__ == "__main__":
    main()
