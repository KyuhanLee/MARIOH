# MARIOH: Multiplicity-Aware Hypergraph Reconstruction

This repository provides the official implementation of MARIOH, a supervised method for reconstructing hyperedges in hypergraphs by leveraging edge multiplicity. MARIOH integrates several key components: a theoretically guaranteed filtering step to identify true size-2 hyperedges, a multiplicity-aware classifier for scoring hyperedge candidates, and a bidirectional search strategy that explores both high- and low-confidence cliques. These components work together to achieve accurate and efficient hypergraph reconstruction. For further details, please refer to our accompanying research paper.

## Contents

- `main.py`: The entry point script to run the hyperedge reconstruction pipeline.
- `params.py`: Parameter dictionaries for various datasets and modes (reduced or preserved).
- `utils/`: A directory containing modularized code for data processing, feature extraction, graph operations, model training, evaluation, and input/output utilities.
- `data/`: A directory that should contain the dataset-specific training and testing files.
  
## Requirements

- **Python Version**: 3.8+ recommended
- **Dependencies**:
  - `numpy`
  - `networkx`
  - `torch`
  - `joblib`
  - `argparse`
  - Additional Python dependencies can be installed via:
    ```bash
    pip install -r requirements.txt
    ```
    
  Adjust `requirements.txt` or the installation commands as needed for your environment.

## Datasets

You must place your datasets into the `data/` directory. Each dataset should have its own subdirectory, for example:

```
data/
戌式式 {dataset_name}/
    戍式 train.txt        # Training data (reduced mode)
    戍式 test.txt         # Testing data (reduced mode)
    戍式 train_dup.txt    # Training data (preserved mode)
    戌式 test_dup.txt     # Testing data (preserved mode)
```

- **Reduced mode** uses `train.txt` and `test.txt`.
- **Preserved mode** uses `train_dup.txt` and `test_dup.txt`.

Please refer to the related publication for details on dataset formats and preprocessing steps.

## Running the Code

To run the pipeline, navigate to the directory containing `main.py` and execute:

```bash
python main.py --data {dataset_name} --gpu 0 --seed 42 --output_dir output
```

### Arguments

- `--data {dataset_name}`: Specify the dataset folder name located under `data/`.
- `--gpu {int}`: GPU device number. If no GPU is available or you wish to run on CPU, set `--gpu` to a non-existent GPU ID (e.g., `--gpu 99`), and it will default to CPU.
- `--seed {int}`: Random seed for reproducibility.
- `--output_dir {path}`: Directory to store the output hyperedge predictions and results.
- `--preserved`: Optional flag. If set, the pipeline will run in "preserved" mode using `train_dup.txt` and `test_dup.txt`. If omitted, the pipeline runs in "reduced" mode using `train.txt` and `test.txt`.

### Examples

**Reduced mode (default):**

```bash
python main.py --data eu --gpu 0 --seed 123 --output_dir output
```

**Preserved mode:**

```bash
python main.py --data eu --gpu 0 --seed 123 --output_dir output --preserved
```

In these examples, the code will:

1. Load and preprocess the graph data.
2. Extract features and prepare a training dataset.
3. Train a classifier network with the best parameters specified in `params.py`.
4. Use the trained classifier to reconstruct hyperedges in the test graph.
5. Save the reconstructed hyperedges to `output/reconstructed_hyp_reduced/{dataset_name}_{seed}.txt` (in reduced mode) or `output/reconstructed_hyp_preserved/{dataset_name}_{seed}.txt` (in preserved mode).

## Interpreting the Results

- **Output Files**: The final reconstructed hyperedges are stored as comma-separated node IDs per line.
- **Evaluation Metrics**: The code prints evaluation metrics such as Jaccard similarity and multiset Jaccard similarity during execution. These metrics help assess the quality of hyperedge reconstruction relative to the ground truth.
- **Performance & Reproducibility**: By setting the random seed (`--seed`) and controlling hyperparameters through `params.py`, you can reproduce experimental results reported in the associated research paper.

## Extending and Customizing

- Modify `params.py` to add or change hyperparameters for different datasets.
- Adjust or add dataset loaders in `utils/data_processing.py` if your input format differs.
- Add new evaluation metrics in `utils/evaluation.py`.
