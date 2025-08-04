import os

# set environment variable to limit the number of threads for scikit-learn
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import torch
import pandas as pd
from survival import Survival_Analysis
from tumor_model import TumorGraphGNN
from tumor_model import TrainerTumorModel
from tumor_model import TumorDataset
from tumor_model import TorchTumorDataset
from cancer_progression import TumorClustering
import utils as Utils

def cluster_patients(
        phylogenies,
        clinical_data,
        weights_path,
        min_label_occurrences,
        node_encoding_type,
        model_hyper_params,
        k_values,
        gamma,
        random_seed,
        device,
        clusterings_save
    ):
    """
    Clusters patients based on their embeddings computed by a trained GNN model.

    Parameters:
    - phylogenies: dictionary of phylogenetic trees for each patient, with patient id as key and list of trees as value.
    - clinical_data: dataframe with clinical data for each patient.
    - weights_path: path to the file storing the weights of the trained GNN-based model.
    - min_label_occurrences: minimum number of occurrences of a mutation in the input dataset to be considered.
    - node_encoding_type: type of node encoding to use: 'clone' or 'mutation'.
    - model_hyper_params: dictionary with hyper parameters used to train the GNN-based model.
    - k_values: list of integers representing the sizes of clusterings to be computed.
    - gamma: multiplicative parameter for the radius of the ball used to filter the outlier embeddings before clustering.
    - event_time: name of the column with survival time in the clinical data.
    - event: name of the column with binary values indicating if death occurred in the clinical data.
    - random_seed: random seed for reproducibility.
    - device: device to use for tensor operations ('cuda', 'cpu' or 'mps').
    - clusterings_save: path where to save the computed clustering results.

    Returns:
    - complete_df: dataframe with both clustering and clinical data.
    """

    # create a TumorDataset object that contains patients sorted by patient id so to allow for reproducibility
    sorted_keys = sorted(phylogenies.keys())
    list_patients = [phylogenies[key] for key in sorted_keys]
    data = TumorDataset(list_patients)

    # compute the set of labels to be considered, based on the number of occurrences in the input dataset
    if min_label_occurrences > 0:
        data.remove_infreq_labels(min_label_occurrences)
    
    # sample one graph per patient
    data.sample_one_graph_per_patient(rd_seed=random_seed)

    # convert the dataset into a TorchTumorDataset object
    torch_data = TorchTumorDataset(data, node_encoding_type=node_encoding_type)

    # get the dataloader for the dataset
    dataloader = TrainerTumorModel.get_dataloader(torch_data, shuffle=False)

    # create a TumorGraphGNN instance with input size based on the labels in the dataset
    model = TumorGraphGNN(
        n_node_labels=len(data.node_labels()),
        h_1=model_hyper_params['h_1'],
        h_2=model_hyper_params['h_2'],
        embedding_dim=model_hyper_params['embedding_dim'],
        dropout_prob_1=model_hyper_params['dropout_prob_1'],
        dropout_prob_2=model_hyper_params['dropout_prob_2'],
        batch_normalization=model_hyper_params['batch_normalization'],
        device=device
    )

    # load the weights of the trained model
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    # compute embeddings for the input dataset
    embeddings = TrainerTumorModel.get_embeddings(model, dataloader, device)

    # filter the embeddings by keeping only those in a ball centered in the mean embedding
    filtered_indices = TumorClustering.ball_filter(embeddings, gamma=gamma)

    # compute the percentage of outliers and print it
    print(f'Percentage of outliers: {TumorClustering.percentage_outliers(embeddings, filtered_indices): .2f}%')

    # compute and save the clustering for the input values of k
    cluster_labels = TumorClustering.cluster_embeddings(embeddings, filtered_indices, k_values=k_values, scale=True)

    # compute the size of each cluster for each value of k and print them
    cluster_sizes = TumorClustering.cluster_sizes_all_k(cluster_labels)
    TumorClustering.print_cluster_sizes_all_k(cluster_sizes)

    # merge in a single dataframe clustering and clinical data
    complete_df = Utils.merge_clustering_clinical_data(cluster_labels, clinical_data)

    # save the computed dataframe with both clustering and clinical data
    os.makedirs(os.path.dirname(clusterings_save), exist_ok=True)
    complete_df.to_csv(clusterings_save, index=False)

    return complete_df

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Computes embeddings for phylogenetic trees associated to patients using a previously trained GNN-based model. Then, it clusters the embeddings and performs a survival analysis on the clusters')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-p', '--phylogenies_path', type=str, required=True, help='Path to the .txt file with pre-processed phylogenetic trees')
    required.add_argument('-c', '--clinical_data_path', type=str, required=True, help='Path to the .csv file with pre-processed clinical data')
    required.add_argument('-w', '--weights', type=str, required=True, help='Path to the file storing the weights of the trained model')

    # optional arguments
    parser.add_argument('--clusterings_save', type=str, default='../results/survival/survival_analysis/breastCancer/clusterings.csv', help='Path where to save the computed clustering results')
    parser.add_argument('--save_plot', type=str, default='../results/survival/survival_analysis/breastCancer/kaplan_meier_curves.pdf', help='Path where to save the Kaplan-Meier curves')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=8, help='Max number of CPU cores for PyTorch')
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the input dataset to be considered')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation"')
    parser.add_argument('--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--h_1', type=int, default=64, help='Output size of the first GCN hidden layer of the trained model')
    parser.add_argument('--h_2', type=int, default=64, help='Output size of the second GCN hidden layer of the trained model')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension of the trained model')
    parser.add_argument('--dropout_prob_1', type=float, default=0.0, help='Dropout probability before the second GCN layer of the trained model')
    parser.add_argument('--dropout_prob_2', type=float, default=0.0, help='Dropout probability before the final linear layer of the trained model')
    parser.add_argument('--batch_normalization', action='store_true', help='Whether batch normalization was used when training the loaded model')
    parser.add_argument('-k', '--k_values', type=int, nargs='+', default=list(range(2, 16)), help='List of values of clustering sizes to be used for clustering. They must be integers greater than 1')
    parser.add_argument('--gamma', type=float, default=1, help='Multiplicative parameter for the radius of the ball used to filter the outlier embeddings before clustering')

    return parser.parse_args()    

if __name__ == '__main__':
    
    # parse command line arguments
    args = parse_args()

    # set the device to use for tensor operations
    device = Utils.get_device(args.device)
    print(f"Using device: {device}")

    # limit the cores used by torch
    torch.set_num_threads(args.max_n_cores)
    torch.set_num_interop_threads(args.max_n_cores)

    # load phylogenetic trees and clinical data
    phylogenies = TrainerTumorModel.load_dataset_txt(args.phylogenies_path)
    clinical_data = pd.read_csv(args.clinical_data_path)

    # hyper parameters used to train the GNN-based model
    model_hyper_params = {
        'h_1': args.h_1,
        'h_2': args.h_2,
        'embedding_dim': args.embedding_dim,
        'dropout_prob_1': args.dropout_prob_1,
        'dropout_prob_2': args.dropout_prob_2,
        'batch_normalization': args.batch_normalization,
    }

    # create the intermediate directories for saving the clustering results, if they do not exist
    os.makedirs(os.path.dirname(args.clusterings_save), exist_ok=True)

    # compute clusterings of patients based on the embeddings computed by a trained GNN-based model
    clusterings_df = cluster_patients(
        phylogenies,
        clinical_data,
        args.weights,
        args.min_label_occurrences,
        args.node_encoding_type,
        model_hyper_params,
        args.k_values,
        args.gamma,
        args.random_seed,
        device,
        args.clusterings_save
    )

    # make a facet figure with a plot for each clustering size and a curve for each cluster in each plot
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    Survival_Analysis.kaplan_meier_plot_for_k(clusterings_df, args.event_time, args.event, 2, args.save_plot)

    # compute and print the log-rank test p-values for each pair of clusters in each clustering
    Survival_Analysis.print_logranks(Survival_Analysis.pairwise_logranks(clusterings_df, args.event_time, args.event))