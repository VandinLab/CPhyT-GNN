import os

# set environment variable to limit the number of threads for scikit-learn
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import json
import copy
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.trial import TrialState
from optuna.importance import get_param_importances
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from functools import partial
from tumor_model import TumorGraphGNN
from tumor_model import TrainerTumorModel as Trainer
from tumor_model import TumorDataset
from tumor_model import TorchTumorDataset
from tumor_model import GraphDistances
from cancer_progression import TumorClustering
from cancer_progression import CloMuClusters
from cancer_progression import Ensemble
from cancer_progression import Baselines
import utils as Utils
from CloMu import trainModel

def load_datasets(datasets_path, node_encoding_type):
    """
    Loads training set and tets set in the format required by CloMu and our model.

    Parameters:
    - datasets_path: string containing the path to the directory storing training and test set in all the formats required by the different methods.
    - node_encoding_type: string indicating the type of node encoding to be used: "clone" or "mutation".

    Returns:
    - train_set: numpy array containing the training set.
    - test_set: numpy array containing the test set.
    - train_data: TumorGraphDataset object containing the training set in the format required by our model.
    - test_data: TumorGraphDataset object containing the test set in the format required by our model.
    - train_torch_data: TorchSurvivalDataset object containing the training set in the format required by our model.
    - test_torch_data: TorchSurvivalDataset object containing the test set in the format required by our model.
    """

    # load the original training and test sets stored in .npy files, as required by CloMu
    train_set = np.load(os.path.join(datasets_path, 'train_set.npy'), allow_pickle=True)
    test_set = np.load(os.path.join(datasets_path, 'test_set.npy'), allow_pickle=True)

    # load training and test data as TumorDataset objects, as required by our model
    train_data = TumorDataset(os.path.join(datasets_path, 'train_set.txt'))
    test_data = TumorDataset(os.path.join(datasets_path, 'test_set.txt'))

    # convert the datasets into TorchTumorDataset objects, using the mapping of node labels computed from the training data also in the test data
    train_torch_data = TorchTumorDataset(train_data, node_encoding_type=node_encoding_type)
    test_torch_data = TorchTumorDataset(test_data, node_encoding_type=node_encoding_type, known_labels_mapping=train_torch_data.node_labels_mapping)

    return train_set, test_set, train_data, test_data, train_torch_data, test_torch_data

def hyperparameters_tuning(optuna_db, random_seed, device, train_torch_data, train_distances, n_train_labels, n_trials):
    """
    Performs a hyper parameters search for our GNN model.

    Parameters:
    - optuna_db: string containing the path to the SQLite database where to save the optimization study.
    - random_seed: random seed for reproducibility.
    - device: device to use.
    - train_torch_data: TorchTumorDataset object containing the training set in the format required by our model.
    - train_distances: tensor with the distances between all pairs of graphs in the training set.
    - n_train_labels: number of labels in the training set.
    - n_trials: number of trials in the hyper parameters optimization search.

    Returns:
    - best_hyperparameters: dictionary containing the best hyperparameters found.
    """

    # create all intermediate directories to the path where the database will be saved, if they do not exist
    os.makedirs(os.path.dirname(optuna_db[len('sqlite:///'):]), exist_ok=True)

    # define the pruner and its parameters
    top_percentile = 70.0                                               # percentile of trials that must perform better than the current trial to be pruned
    n_startup_trials = 10                                               # number of complete trials that must be performed before the pruner starts to prune
    n_warmup_steps = 10                                                 # number of epochs that must be performed before the pruner starts to prune a trial
    patience = 5                                                        # number of epochs representing the pruning patience
    base_pruner = optuna.pruners.PercentilePruner(percentile=top_percentile, n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    composed_pruner = optuna.pruners.PatientPruner(base_pruner, patience=patience)

    # create an optuna study
    study = optuna.create_study(
        storage=optuna_db,
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        direction='minimize',
        study_name=f'tumor_model_tuning_{random_seed}',
        pruner=composed_pruner
    )
    
    # optimize the objective function
    study.optimize(
        partial(
            Trainer.tuning_objective,
            train_torch_data=train_torch_data,
            val_torch_data=train_torch_data,                            # we optimize the hyper parameters based on the training loss
            train_distances=train_distances,
            val_distances=train_distances,
            n_labels=n_train_labels,
            random_seed=random_seed,
            device=device
        ),
        n_trials=n_trials,
        n_jobs=1
    )

    # report some statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("\nStudy statistics: ")
    print(f"number of finished trials: {len(study.trials)}")
    print(f"number of pruned trials: {len(pruned_trials)}")
    print(f"number of complete trials: {len(complete_trials)}")

    # extract the best trial
    best_trial = study.best_trial

    # extract and print the best configuration of hyperparameters
    best_hyperparameters = study.best_trial.params
    print('\nBest trial hyperparameters:')
    for key, value in best_hyperparameters.items():
        print(f'{key}: {value}')
    
    # print the score of the best trial
    print(f"\nScore of the best trial: {best_trial.value}")

    # compute the importance of the hyperparameters and print it
    hyperparameter_importances = get_param_importances(study)
    print("\nHyperparameter importances:")
    for key, value in hyperparameter_importances.items():
        print(f'{key}: {value}')

    return best_hyperparameters

def cluster_GNN_embeddings(embeddings, k_values, gamma, clusterings_dir):
    """
    Clusters the imput embeddings using the input values of clustering size.
    It uses k-means considering the Euclidean distance between embeddings.

    Parameters:
    - embeddings: torch tensor containing the embeddings to be clustered.
    - k_values: list of values of k to be used for clustering.
    - gamma: multiplicative parameter for the radius of the ball used to filter the outlier embeddings before clustering.
    - clusterings_dir: path to the base directory where to save the clustering results.

    Returns:
    - clusterings_labels: dictionary containing the cluster labels for each value of k.
    """

    # filter the embeddings by keeping only those in a ball centered in the mean embedding
    filtered_indices = TumorClustering.ball_filter(embeddings, gamma=gamma)

    # compute the percentage of outliers and print it
    print(f'Percentage of outliers: {TumorClustering.percentage_outliers(embeddings, filtered_indices): .2f}%')

    # compute and save the clustering for the input values of k
    clusterings_labels = TumorClustering.cluster_embeddings(embeddings, filtered_indices, k_values=k_values, scale=True)

    # compute the size of each cluster for each value of k and print them
    cluster_sizes = TumorClustering.cluster_sizes_all_k(clusterings_labels)
    TumorClustering.print_cluster_sizes_all_k(cluster_sizes)

    # create the folder where to save the computed cluster labels, if it does not exist
    os.makedirs(clusterings_dir, exist_ok=True)

    # save the computed cluster labels, with one file per value of k
    TumorClustering.save_cluster_labels(clusterings_labels, clusterings_dir)

    return clusterings_labels

def cluster_oncotree2vec_embeddings(embeddings, k_values, clusterings_dir):
    """
    Clusters the imput embeddings using the input values of clustering size.
    It uses hierarchical clustering considering the cosine distance between embeddings, as done by the authors of oncotree2vec.

    Parameters:
    - embeddings: numpy array containing the embeddings to be clustered. It has shape (n_patients, n_features).
    - k_values: list of values of k to be used for clustering.
    - clusterings_dir: path to the base directory where to save the clustering results.

    Returns:
    - clusterings_labels: dictionary containing the cluster labels for each value of k.
    """

    # dictionary that will contain the cluster labels for each input value of k
    clusterings_labels = {}

    # dictionary that will contain the silhouette score for each input value of k
    silhouette_clusterings = {}

    # scale the embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # compute a clustering for each input value of k
    for k in tqdm(k_values, desc='Computing clusterings', unit='clg'):
        hierarchical_clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        clusterings_labels[k] = hierarchical_clustering.fit_predict(embeddings)
        silhouette_clusterings[k] = silhouette_score(embeddings, clusterings_labels[k])

    # print the average silhouette score for each computed clustering
    for k, s in silhouette_clusterings.items():
        print(f'K = {k}: s = {s}')

    # compute the size of each cluster for each value of k and print them
    clusterings_sizes = TumorClustering.cluster_sizes_all_k(clusterings_labels)
    TumorClustering.print_cluster_sizes_all_k(clusterings_sizes)

    # create the folder where to save the computed cluster labels, if it does not exist
    os.makedirs(clusterings_dir, exist_ok=True)

    # save the computed cluster labels, with one file per value of k
    TumorClustering.save_cluster_labels(clusterings_labels, clusterings_dir)

    return clusterings_labels

def create_cluster_data_for_CloMu(train_set, test_set, clusterings_labels, clusterings_dir):
    """
    Saves each cluster in each clustering as a separate .npy file with the test set appended at the end, as required by CloMu.

    Parameters:
    - train_set: numpy array containing the training set. Each element is a patient and each patient is a list of phylogenetic trees.
    - test_set: numpy array containing the test set. Each element is a patient and each patient is a list of phylogenetic trees.
    - clusterings_labels: dictionary containing the cluster labels for each value of k.
    - clusterings_dir: path to the base directory where to save the clustering results.
    """

    # iterate though the clusterings
    for k, cluster_labels in clusterings_labels.items():

        # directory where to save the current clustering
        curr_dir = os.path.join(clusterings_dir, f'k_{k}')

        # assign each patient to its computed cluster
        clusters = CloMuClusters.cluster_patients(train_set, cluster_labels)

        # save clusters followed by the test set so to be able to run CloMu then
        CloMuClusters.save_clusters_and_test(clusters, test_set, curr_dir)

def random_clustering(train_set, test_set, k_values, GNN_clusterings_dir, random_clusterings_dir):
    """
    Clusters the input training set randomly, but so to have clusters with the same number of patients as the clusters produced by the GNN method.

    Parameters:
    - train_set: numpy array containing the training set. Each element is a patient and each patient is a list of phylogenetic trees.
    - test_set: numpy array containing the test set. Each element is a patient and each patient is a list of phylogenetic trees.
    - k_values: list of clustering sizes.
    - GNN_clusterings_dir: path to the folder containing the clustering produced by the GNN method for each value of k.
    - random_clusterings_dir: path where to save the random clusterings.
    """

    # iterate though the clusterings
    for k in k_values:

        # path to the folder containing the clustering produced by the GNN method for the current value of k
        GNN_clusters_path = os.path.join(GNN_clusterings_dir, f'k_{k}')

        # path where to save the current clustering
        curr_random_clusters_path = os.path.join(random_clusterings_dir, f'k_{k}')
        os.makedirs(curr_random_clusters_path, exist_ok=True)

        # compute the number of patients in each cluster
        n_patients_clusters = Baselines.clusters_dimension(GNN_clusters_path, k, test_set)
        
        # cluster the training patients randomly, but so to have clusters with the desired number of patients
        rd_clustering = Baselines.random_clustering(train_set, n_patients_clusters)

        # save the computed clusters with the test set appended at the end, as required by CloMu
        for i in range(k):
            np.save(os.path.join(curr_random_clusters_path, f'cluster_{i}.npy'), np.concatenate((rd_clustering[i], test_set), axis=0))

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Run our GNN-based model on a training set of phylogenetic trees, then cluster patients based on the embeddings and train a CloMu instance on each cluster. Finally, cluster patients using other baseline methods')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--datasets_dir', type=str, required=True, help='Path to the directory containing all the input data required for the experiment. In particular, there must be training and test set in .npy format and training and test in the formats required by all the considered methods.' \
                          'If RECAP is included in the experiment, then the directory must contain also the clusterings produced by RECAP for all the input clustering sizes. They must be in a subdirectory of the provided path, named "clusterings_RECAP".' \
                          'Inside "clusterings_RECAP" there must be a file "$s_clusters.solution.txt" for each clustering of size $s produced by RECAP.' \
                          'If RECAP_f is included in the experiment, then the directory must contain also the clusterings produced by RECAP with the option "-c $f" for all the input clustering sizes. They must be in a subdirectory of the provided path, named "clusterings_RECAP_$f".' \
                          'Inside "clusterings_RECAP_$f" there must be a file "$s_clusters.solution.txt" for each clustering of size $s produced by RECAP_f.' \
                          'If oncotree2vec is included in the experiment, then the directory must contain also the embeddings computed by oncotree2vec in a file named "embeddings_oncotree2vec.csv" such that the rows in the .csv file are aligned with the numpy array of patients in the .npy file storing the training set')

    # optional arguments
    parser.add_argument('-d', '--embedding_dim', type=int, default=64, help='Size of the embedding produced by our GNN-based model. For getting the results in the paper, we set it to 64 for the breast cancer dataset and to 32 for the AML dataset. This parameter is used only if tuning is not performed')
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=4, help='Max number of CPU cores for PyTorch')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation"')
    parser.add_argument('--max_tree_length', type=int, default=9, help='Maximum length in terms of edges of a phylogenetic tree to be considered')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials in the hyper parameters optimization search')
    parser.add_argument('--save_plot', action='store_true', help='Saves a plot reporting the training loss over epochs')
    parser.add_argument('--verbose', action='store_true', help='Print training information during training')
    parser.add_argument('--save_best_params', action='store_true', help='Saves the best hyper parameters found in the optimization search')
    parser.add_argument('-k', '--k_values', type=int, nargs='+', default=[2, 3, 4], help='List of values of clustering sizes to be used for clustering. They must be integers greater than 1')
    parser.add_argument('--gamma', type=float, default=1, help='Multiplicative parameter for the radius of the ball used to filter the outlier embeddings before clustering')
    parser.add_argument('--infinite_sites', action='store_true', help='If set, the infinite sites assumption is enabled for CloMu. This option must be included when considering the BreastCancer.npy dataset')
    parser.add_argument('--CloMu_epochs', type=int, default=500, help='Number of epochs for the training of CloMu on each cluster')
    parser.add_argument('--no_tuning', action='store_true', help='If set, our GNN-based model is not tuned before training. In this case, the best hyperparameters are set to default ones')
    parser.add_argument('--no_GNN', action='store_true', help='If set, our GNN-based model is not considered in the experiment. In this case, also the random baseline, which depends on the GNN-based clustering, is automatically not considered')
    parser.add_argument('--no_random', action='store_true', help='If set, the random baseline is not considered in the experiment')
    parser.add_argument('--no_CloMu_based', action='store_true', help='If set, the CloMu-based baseline is not considered in the experiment')
    parser.add_argument('--no_standard_CloMu', action='store_true', help='If set, the standard CloMu baseline is not considered in the experiment')
    parser.add_argument('--no_RECAP', action='store_true', help='If set, RECAP is not considered for the experiment. To include RECAP, it must be run before this script on the same training set')
    parser.add_argument('--no_oncotree2vec', action='store_true', help='If set, oncotree2vec is not considered in the experiment. To include oncotree2vec, it must be run before this script on the same training set')
    parser.add_argument('-f', '--min_label_occurrences', type=int, default=100, help='Minimum number of occurrences of a label in the training set for it to be considered during the training of our GNN_f model, RECAP_f and Random_f with only frequent mutations, if enabled')
    parser.add_argument('--no_GNN_f', action='store_true', help='If set, the GNN-based model with the input f is not considered in the experiment. In this case, also the random baseline with the same f, which depends on the GNN-based clustering, is automatically not considered')
    parser.add_argument('--no_random_f', action='store_true', help='If set, the random baseline with the input f is not considered in the experiment')
    parser.add_argument('--no_RECAP_f', action='store_true', help='If set, RECAP with the input f is not considered in the experiment. To include RECAP_f, it must be run before this script on the same training set')

    return parser.parse_args()    

if __name__ == '__main__':

    # ------------------------------------------------------ LOAD DATA ------------------------------------------------------

    # print information
    print('\nLoading data...\n')

    # parse the command line arguments
    args = parse_args()

    # set the device
    device = Utils.get_device(args.device)

    # limit the cores used by torch
    torch.set_num_threads(args.max_n_cores)
    torch.set_num_interop_threads(args.max_n_cores)

    # load training and test sets in the formats required by CloMu and our model
    train_set_np, test_set_np, train_data, test_data, train_torch_data, test_torch_data = load_datasets(args.datasets_dir, node_encoding_type=args.node_encoding_type)

    # get the paths to training, test and train_test datasets in .npy format, required by CloMu
    train_set_path = os.path.join(args.datasets_dir, 'train_set.npy')
    test_set_path = os.path.join(args.datasets_dir, 'test_set.npy')
    train_test_path = os.path.join(args.datasets_dir, 'train_test.npy')

    # ------------------------------------------------------ GNN-BASED MODEL ------------------------------------------------------

    # include our GNN-based model in the experiments only if not excluded by the user
    if not args.no_GNN:

        # compute the tensor with the distances between all pairs of graphs in the training set
        train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

        # path to the directory where to store all the results for the GNN-based model
        GNN_dir_path = os.path.join(args.datasets_dir, '..', 'GNN')
        os.makedirs(GNN_dir_path, exist_ok=True)

        # set the hyper parameters of the GNN-based model to default
        best_hyperparameters = {
            'weight_decay': 0.0,
            'batch_size': 64,
            'loss_fn': 'MAE_loss',
            'optimizer': 'Adam',
            'lr': 1e-3,
            'epochs': 500,
            'batch_normalization': False,
            'dropout_prob_1': 0.0,
            'dropout_prob_2': 0.0,
            'h_1': 64,
            'h_2': 64,
            'embedding_dim': args.embedding_dim
        }

        # tune the hyper parameters of the GNN-based model only if the user did not set the flag to skip the tuning
        if not args.no_tuning:

            # print information
            print('\nTuning GNN-based model...\n')

            # path where to save the SQLite database with the optimization study
            optuna_db = os.path.join('sqlite:///', GNN_dir_path, 'model_tuning.sqlite3')

            # find the best hyperparameters for the GNN model based on the training loss
            best_hyperparameters = hyperparameters_tuning(
                optuna_db,
                args.random_seed,
                device,
                train_torch_data,
                train_distances,
                len(train_data.node_labels()),
                args.n_trials
            )

            # save the best hyperparameters to a file, if required
            if args.save_best_params:
                best_params_save_path = os.path.join(GNN_dir_path, 'best_hyperparameters.json')
                with open(best_params_save_path, 'w') as file:
                    json.dump(best_hyperparameters, file, indent=4) 

        # print information
        if not args.no_tuning:
            print('\nTraining GNN-based model with the best found hyper parameters...\n')
        else:
            print('\nTraining GNN-based model with default hyper parameters...\n')

        # path where to save the weights of the trained model
        weights_path = os.path.join(GNN_dir_path, 'weights.npy')

        # create a model instance
        model = TumorGraphGNN(
            n_node_labels=len(train_data.node_labels()),
            h_1=best_hyperparameters['h_1'],
            h_2=best_hyperparameters['h_2'],
            embedding_dim=best_hyperparameters['embedding_dim'],
            dropout_prob_1=best_hyperparameters['dropout_prob_1'],
            dropout_prob_2=best_hyperparameters['dropout_prob_2'],
            batch_normalization=best_hyperparameters['batch_normalization'],
            device=device
        )

        # if required, save the training plot
        plot_save = None
        if args.save_plot:
            plot_save = os.path.join(GNN_dir_path, 'training_plot.jpg')

        # train the model instance on the data with the found best hyperparameters
        Trainer.train(
            model,
            train_torch_data,
            train_distances,
            loss_fn=Utils.select_loss(best_hyperparameters['loss_fn']),
            optimizer=Utils.select_optimizer(best_hyperparameters['optimizer']),
            weight_decay=best_hyperparameters['weight_decay'],
            batch_size=best_hyperparameters['batch_size'],
            val_data=None,
            val_graph_distances=None,
            plot_save=plot_save,
            verbose=args.verbose,
            epochs=best_hyperparameters['epochs'],
            lr=best_hyperparameters['lr'],
            early_stopping_tolerance=None,
            save_model=weights_path,
            device=device
        )

        # print information
        print('\nComputing embeddings for training patients using our GNN-based model...\n')

        # create a dataloader storing the training phylogenetic trees
        train_dataloader = Trainer.get_dataloader(train_torch_data, batch_size=best_hyperparameters['batch_size'], shuffle=False)

        # compute the embeddings for the phylogenetic trees in the training set
        embeddings = Trainer.get_embeddings(model, train_dataloader, device=device)

        # print information
        print('\nClustering training patients using the embeddings computed by the GNN-based model...\n')

        # compute the cluster labels for the training embeddings for all the input clustering sizes
        clusterings_labels = cluster_GNN_embeddings(embeddings, args.k_values, args.gamma, GNN_dir_path)

        # prepare the clustered training set to be fed to CloMu
        create_cluster_data_for_CloMu(train_set_np, test_set_np, clusterings_labels, GNN_dir_path)

        # print information
        print('\nTraining a different CloMu instance on each cluster in each clustering computed using our GNN-based model...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder with clusters
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu weights for the clusters
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )

    # ------------------------------------------------------ RANDOM BASELINE ------------------------------------------------------

    # include the random clustering baseline in the experiments only if both the GNN-based model and the random baseline are not excluded by the user
    if not args.no_GNN and not args.no_random:

        # path to the directory where to store all the results for the random baseline
        random_dir_path = os.path.join(args.datasets_dir, '..', 'Random')
        os.makedirs(random_dir_path, exist_ok=True)        

        # print information
        print('\nClustering the training patients at random, but into clusters of the same sizes of those computed by our GNN-based model...\n')

        # randomly cluster the training set, but so to have clusters with the same number of patients as the clusters produced by the GNN method
        random_clustering(train_set_np, test_set_np, args.k_values, GNN_dir_path, random_dir_path)

        # print information
        print('\nTraining a different CloMu instance on each cluster in each clustering computed ar random...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder with clusters
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder where to save CloMu weights for the clusters
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )
        
    # ------------------------------------------------------ STANDARD CLOMU BASELINE ------------------------------------------------------

    # include the standard CloMu baseline in the experiments only if not excluded by the user
    if not args.no_standard_CloMu:

        # path to the directory where to store all the results for the standard CloMu model
        standard_CloMu_dir_path = os.path.join(args.datasets_dir, '..', 'Standard_CloMu')
        os.makedirs(standard_CloMu_dir_path, exist_ok=True)        

        # print information
        print('\nTraining a single CloMu instance on the whole training set...\n')

        # train a CloMu model instance only on training patients, making it infer probabilities also for test trees after training
        trainModel(
            [train_test_path],                                                                                                           # list of datasets with both training and test set
            os.path.join(standard_CloMu_dir_path, f'Standard_CloMu.pth'),                                                                # path where to save the trained CloMu weights
            os.path.join(standard_CloMu_dir_path, f'Standard_CloMu_probabilities.npy'),                                                  # path where to save the probabilities assigned both to train and test trees
            os.path.join(standard_CloMu_dir_path, f'Standard_CloMu_mutations.npy'),                                                      # path where to save the mutation names found during training
            patientNames='',
            inputFormat='raw',                                                                                                           # format of the input dataset
            infiniteSites=args.infinite_sites,                                                                                           # whether the infinite sites assumption must be enabled or not
            trainSize=train_set_np.shape[0],                                                                                             # number of patients in the training set
            maxM=args.max_tree_length,                                                                                                   # maximum length of a tree to be considered, otherwise it is removed            
            regularizeFactor='default',                                                                                                  # regularize factor  
            iterations=args.CloMu_epochs,                                                                                                # number of training epochs
            verbose=True                                                                                                                 # whether to print information during training
        )

    # ------------------------------------------------------ CLOMU-BASED BASELINE ------------------------------------------------------

    # include the CloMu-based baseline in the experiments only if not excluded by the user
    if not args.no_CloMu_based:

        # path to the directory where to store all the results for the CloMu-based model
        CloMu_based_dir_path = os.path.join(args.datasets_dir, '..', 'CloMu_based')
        os.makedirs(CloMu_based_dir_path, exist_ok=True)        

        # print information
        print('\nClustering training patients using the CloMu-based baseline...\n')

        # compute and save clusterings of the training set for all input sizes
        Baselines.balanced_CloMu_probabilities_clustering(train_set_path, CloMu_based_dir_path, args.k_values, test_set_np, args.infinite_sites, args.max_tree_length, args.CloMu_epochs)

        # print information
        print('\nTraining a different CloMu instance on each cluster in each clustering computed using CloMu-based baseline...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(CloMu_based_dir_path, f'k_{k}'),                                                 # path to the folder with clusters
                os.path.join(CloMu_based_dir_path, f'k_{k}'),                                                 # path to the folder where to save CloMu weights for the clusters
                os.path.join(CloMu_based_dir_path, f'k_{k}'),                                                 # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(CloMu_based_dir_path, f'k_{k}'),                                                 # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )

    # ------------------------------------------------------ RECAP BASELINE ------------------------------------------------------

    # include RECAP in the experiments only if not excluded by the user
    if not args.no_RECAP:

        # print information
        print('\nLoading RECAP clusterings...\n')

        # path to the directory where the clusterings computed by RECAP are stored
        RECAP_clusters_path = os.path.join(args.datasets_dir, 'clusterings_RECAP')

        # path to the directory where to store all the results for RECAP
        RECAP_dir_path = os.path.join(args.datasets_dir, '..', 'RECAP')
        os.makedirs(RECAP_dir_path, exist_ok=True)

        # load the clustering labels produced by RECAP for all the input clustering sizes
        clusterings_labels = {}
        for k in args.k_values:
            clusterings_labels[k] = CloMuClusters.RECAP_extract_cluster_indices(os.path.join(RECAP_clusters_path, f'{k}_clusters.solution.txt'))

        # prepare the clustered training set to be fed to CloMu
        create_cluster_data_for_CloMu(train_set_np, test_set_np, clusterings_labels, RECAP_dir_path)

        # print information
        print('\nTraining a different CloMu instance on each cluster in each clustering computed by RECAP...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder with clusters
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder where to save CloMu weights for the clusters
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )
    # ------------------------------------------------------ ONCOTREE2VEC BASELINE ------------------------------------------------------

    # include oncotree2vec in the experiments only if not excluded by the user
    if not args.no_oncotree2vec:

        # print information
        print('\nClustering the embeddings for patients computed by oncotree2vec...\n')

        # path to the directory where to store all the results for oncotree2vec
        oncotree2vec_dir_path = os.path.join(args.datasets_dir, '..', 'oncotree2vec')
        os.makedirs(oncotree2vec_dir_path, exist_ok=True)

        # load the embeddings computed by oncotree2vec
        oncotree2vec_embeddings = pd.read_csv(os.path.join(args.datasets_dir, 'embeddings_oncotree2vec.csv'), index_col=0)

        # convert the embeddings to a numpy array
        oncotree2vec_embeddings = oncotree2vec_embeddings.to_numpy()

        # cluster the embeddings for each input clustering size
        clusterings_labels = cluster_oncotree2vec_embeddings(oncotree2vec_embeddings, args.k_values, oncotree2vec_dir_path)

        # prepare the clustered training set to be fed to CloMu
        create_cluster_data_for_CloMu(train_set_np, test_set_np, clusterings_labels, oncotree2vec_dir_path)

        # print information
        print('\nTraining a different CloMu instance on each cluster in each clustering computed applying hierarchical clustering with cosine distance to oncotree2vec embeddings...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(oncotree2vec_dir_path, f'k_{k}'),                                                         # path to the folder with clusters
                os.path.join(oncotree2vec_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu weights for the clusters
                os.path.join(oncotree2vec_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(oncotree2vec_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )
    
    # ------------------------------------------------------ GNN-BASED MODEL WITH INPUT F ------------------------------------------------------
    
    # include GNN_f in the experiments only if not excluded by the user
    if not args.no_GNN_f:

        # minimum number of occurrences of a label in the training set for it to be considered during the training of our GNN_f model
        f = args.min_label_occurrences

        # create training data with only frequent mutations
        train_data.remove_infreq_labels(f)
        train_torch_data = TorchTumorDataset(train_data, node_encoding_type=args.node_encoding_type)
        train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

        # path to the directory where to store all the results for GNN_f
        GNN_dir_path = os.path.join(args.datasets_dir, '..', f'GNN_{f}')
        os.makedirs(GNN_dir_path, exist_ok=True)

        # set the hyper parameters of GNN_f to default
        best_hyperparameters = {
            'weight_decay': 0.0,
            'batch_size': 64,
            'loss_fn': 'MAE_loss',
            'optimizer': 'Adam',
            'lr': 1e-3,
            'epochs': 500,
            'batch_normalization': False,
            'dropout_prob_1': 0.0,
            'dropout_prob_2': 0.0,
            'h_1': 64,
            'h_2': 64,
            'embedding_dim': args.embedding_dim
        }

        # tune the hyper parameters of GNN_f only if the user did not set the flag to skip the tuning
        if not args.no_tuning:

            # print information
            print(f'\nTuning GNN_{f}...\n')

            # path where to save the SQLite database with the optimization study
            optuna_db = os.path.join('sqlite:///', GNN_dir_path, 'model_tuning.sqlite3')

            # find the best hyperparameters for the GNN model based on the training loss
            best_hyperparameters = hyperparameters_tuning(
                optuna_db,
                args.random_seed,
                device,
                train_torch_data,
                train_distances,
                len(train_data.node_labels()),
                args.n_trials
            )

            # save the best hyperparameters to a file, if required
            if args.save_best_params:
                best_params_save_path = os.path.join(GNN_dir_path, 'best_hyperparameters.json')
                with open(best_params_save_path, 'w') as file:
                    json.dump(best_hyperparameters, file, indent=4) 

        # print information
        if not args.no_tuning:
            print(f'\nTraining GNN_{f} with the best found hyper parameters...\n')
        else:
            print(f'\nTraining GNN_{f} with default hyper parameters...\n')

        # path where to save the weights of the trained model
        weights_path = os.path.join(GNN_dir_path, 'weights.npy')

        # create a model instance
        model = TumorGraphGNN(
            n_node_labels=len(train_data.node_labels()),
            h_1=best_hyperparameters['h_1'],
            h_2=best_hyperparameters['h_2'],
            embedding_dim=best_hyperparameters['embedding_dim'],
            dropout_prob_1=best_hyperparameters['dropout_prob_1'],
            dropout_prob_2=best_hyperparameters['dropout_prob_2'],
            batch_normalization=best_hyperparameters['batch_normalization'],
            device=device
        )

        # if required, save the training plot
        plot_save = None
        if args.save_plot:
            plot_save = os.path.join(GNN_dir_path, 'training_plot.jpg')

        # train the model instance on the data with the found best hyperparameters
        Trainer.train(
            model,
            train_torch_data,
            train_distances,
            loss_fn=Utils.select_loss(best_hyperparameters['loss_fn']),
            optimizer=Utils.select_optimizer(best_hyperparameters['optimizer']),
            weight_decay=best_hyperparameters['weight_decay'],
            batch_size=best_hyperparameters['batch_size'],
            val_data=None,
            val_graph_distances=None,
            plot_save=plot_save,
            verbose=args.verbose,
            epochs=best_hyperparameters['epochs'],
            lr=best_hyperparameters['lr'],
            early_stopping_tolerance=None,
            save_model=weights_path,
            device=device
        )

        # print information
        print(f'\nComputing embeddings for training patients using GNN_{f}...\n')

        # create a dataloader storing the training phylogenetic trees
        train_dataloader = Trainer.get_dataloader(train_torch_data, batch_size=best_hyperparameters['batch_size'], shuffle=False)

        # compute the embeddings for the phylogenetic trees in the training set
        embeddings = Trainer.get_embeddings(model, train_dataloader, device=device)

        # print information
        print(f'\nClustering training patients using the embeddings computed by GNN_{f}...\n')

        # compute the cluster labels for the training embeddings for all the input clustering sizes
        clusterings_labels = cluster_GNN_embeddings(embeddings, args.k_values, args.gamma, GNN_dir_path)

        # prepare the clustered training set to be fed to CloMu
        create_cluster_data_for_CloMu(train_set_np, test_set_np, clusterings_labels, GNN_dir_path)

        # print information
        print(f'\nTraining a different CloMu instance on each cluster in each clustering computed using GNN_{f}...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder with clusters
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu weights for the clusters
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(GNN_dir_path, f'k_{k}'),                                                         # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )

    # ------------------------------------------------------ RANDOM BASELINE WITH INPUT F ------------------------------------------------------

    # include the random clustering baseline with input f in the experiments only if both GNN_f and random_f baseline are not excluded by the user
    if not args.no_GNN_f and not args.no_random_f:

        # path to the directory where to store all the results for the random baseline
        random_dir_path = os.path.join(args.datasets_dir, '..', f'Random_{f}')
        os.makedirs(random_dir_path, exist_ok=True)        

        # print information
        print(f'\nClustering the training patients at random, but into clusters of the same sizes of those computed by GNN_{f} model...\n')

        # randomly cluster the training set, but so to have clusters with the same number of patients as the clusters produced by the GNN method
        random_clustering(train_set_np, test_set_np, args.k_values, GNN_dir_path, random_dir_path)

        # print information
        print(f'\nTraining a different CloMu instance on each cluster in each clustering computed by random_{f}...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder with clusters
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder where to save CloMu weights for the clusters
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(random_dir_path, f'k_{k}'),                                                      # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )

    # ------------------------------------------------------ RECAP BASELINE WITH INPUT F ------------------------------------------------------

    # include RECAP with input f in the experiments only if not excluded by the user
    if not args.no_RECAP_f:

        # minimum number of occurrences of a label in the training set considered for the training of RECAP_f
        f = args.min_label_occurrences
        
        # print information
        print(f'\nLoading RECAP_{f} clusterings...\n')

        # path to the directory where the clusterings computed by RECAP_f are stored
        RECAP_clusters_path = os.path.join(args.datasets_dir, f'clusterings_RECAP_{f}')

        # path to the directory where to store all the results for RECAP_f
        RECAP_dir_path = os.path.join(args.datasets_dir, '..', f'RECAP_{f}')
        os.makedirs(RECAP_dir_path, exist_ok=True)

        # load the clustering labels produced by RECAP_f for all the input clustering sizes
        clusterings_labels = {}
        for k in args.k_values:
            clusterings_labels[k] = CloMuClusters.RECAP_extract_cluster_indices(os.path.join(RECAP_clusters_path, f'{k}_clusters.solution.txt'))

        # prepare the clustered training set to be fed to CloMu
        create_cluster_data_for_CloMu(train_set_np, test_set_np, clusterings_labels, RECAP_dir_path)

        # print information
        print(f'\nTraining a different CloMu instance on each cluster in each clustering computed by RECAP_{f}...\n')

        # train a CloMu model on each cluster of each clustering and compute probabilities also for test trees, saving all the results
        for k in args.k_values:
            Ensemble.train_on_clusters(
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder with clusters
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder where to save CloMu weights for the clusters
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder where to save CloMu probabilities for the clusters
                os.path.join(RECAP_dir_path, f'k_{k}'),                                                       # path to the folder where to save CloMu mutations for the clusters
                test_set_path,                                                                                # path to the test set concatenated to all clusters
                k,                                                                                            # number of clusters
                'raw',                                                                                        # format of the input dataset
                args.infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                args.max_tree_length,                                                                         # maximum length of a tree to be considered, otherwise it is removed            
                'default',                                                                                    # regularize factor
                args.CloMu_epochs                                                                             # number of training epochs
            )