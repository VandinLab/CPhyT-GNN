import argparse
import os
import numpy as np
import networkx as nx
from tumor_model import TumorDataset
from tumor_model import TorchTumorDataset
import utils as Utils

def pre_processing(dataset_path, max_tree_length, train_ratio, random_seed, train_set_path, test_set_path, train_test_path, gene_level_analysis=False, min_label_occurrences=0):
    """
    Pre-processes the input dataset and splits it inot training and test set, returning them both in the format required by CloMu and in the format required by our model.

    Parameters:
    - dataset_path: string containing the path to the .npy file with the complete dataset.
    - max_tree_length: maximum length in terms of edges of a phylogenetic tree to be considered.
    - train_ratio: ratio of patients in the input dataset to be included in the training set. The other are included in the test set.
    - random_seed: random seed for reproducibility.
    - train_set_path: path to the .npy file where to save the training set.
    - test_set_path: path to the .npy file where to save the test set.
    - train_test_path: path to the .npy file where to save the training set and test set concatenated together.
    - gene_level_analysis: if set, collapse mutations at gene level.
    - min_label_occurrences: minimum number of occurrences of a mutation in the input dataset to be considered.

    Returns:
    - train_set: numpy array containing the training set.
    - test_set: numpy array containing the test set.
    - train_data: TumorGraphDataset object containing the training set in the format required by our model.
    - test_data: TumorGraphDataset object containing the test set in the format required by our model.
    """

    # load the original dataset as a numpy array
    dataset = np.load(dataset_path, allow_pickle=True)

    # remove trees that are too long, as done by the authors of CloMu
    processed_dataset = Utils.remove_long_trees(dataset, max_length=max_tree_length)

    # split the dataset into a random training set and a random test set, using the input random seed for reproducibility
    train_set, test_set = Utils.train_test_split(processed_dataset, p=train_ratio, rd_seed=random_seed)

    # save the training set, test set and a single .npy file with the test set appended at the end of the training set
    np.save(train_set_path, train_set)
    np.save(test_set_path, test_set)
    np.save(train_test_path, np.concatenate((train_set, test_set), axis=0))

    # convert each dataset into the format required by our model
    train_phylogenies_dic = Utils.convert_into_dictionaries(train_set)
    test_phylogenies_dic = Utils.convert_into_dictionaries(test_set)
    
    # if a gene-level analysis is required, then collapse mutations at gene level
    if gene_level_analysis:
        train_phylogenies_dic = Utils.collapse_gene_level(train_phylogenies_dic)
        test_phylogenies_dic = Utils.collapse_gene_level(test_phylogenies_dic)

    # save training and test sets into a .txt file with the format required by our model
    Utils.save_dataset_txt(train_phylogenies_dic, train_set_path[:-len('.npy')] + '.txt')
    Utils.save_dataset_txt(test_phylogenies_dic, test_set_path[:-len('.npy')] + '.txt')

    # create TumorDataset objects for training and test sets
    train_data = TumorDataset(train_set_path[:-len('.npy')] + '.txt')
    test_data = TumorDataset(test_set_path[:-len('.npy')] + '.txt')

    # compute the set of labels to be considered, based on the number of occurrences in the training set
    if min_label_occurrences > 0:
        train_data.remove_infreq_labels(min_label_occurrences)
        test_data.replace_label_set(train_data.node_labels(), replace_with='empty')

    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=random_seed)
    test_data.sample_one_graph_per_patient(rd_seed=random_seed)

    # save the pre-processed training set and test set into a .txt file with the format required by our model
    train_data.save_dataset(train_set_path[:-len('.npy')] + '.txt')
    test_data.save_dataset(test_set_path[:-len('.npy')] + '.txt')

    return train_set, test_set, train_data, test_data

def convert_to_RECAP(dataset, path):
    """
    Saves a dataset stored as an numpy array in a .txt file following the format required by RECAP.

    Parameters:
    - dataset: numpy array containing patients stored as lists of trees, where each tree is a list of edges.
    - path: path where to store the converted dataset.
    """

    # open the file
    with open(path, 'w') as file:
        
        # the first thing to write is the number of patients
        first_line = f'{dataset.shape[0]} # patients\n'
        file.write(first_line)

        # iterate over patients
        for i in range(dataset.shape[0]):

            # write the number of trees contained by the current patient
            curr_line = f'{len(dataset[i])} # trees patient {i}\n'
            file.write(curr_line)

            # iterate over the trees of the current patient
            for j in range(len(dataset[i])):

                # write the number of edges in the current tree
                curr_line = f'{len(dataset[i][j])} # edges tree {j}\n'
                file.write(curr_line)

                # iterate over the edges of the current tree
                for edge in dataset[i][j]:

                    # save the two ordered mutations related to the current edge
                    mut_1 = edge[0]
                    mut_2 = edge[1]

                    # write the two ordered mutations on a line of the file
                    curr_line = f'{mut_1} {mut_2}\n'
                    file.write(curr_line)

def convert_to_oncotree2vec(tumor_data, path):
    """
    Converts the input dataset into a format suitable for oncotree2vec, saving each phylogenetic tree as a .gexf file.

    Parameters:
    - tumor_data: TumorDataset object containing the dataset to be converted.
    - path: path to the directory where to save the files with the converted phylogenetic trees.
    """

    # map the mutation labels to integers
    node_labels = tumor_data.node_labels()
    labels_mapping = TorchTumorDataset.map_node_labels(node_labels)

    # convert each TumorGraph into a NetworkX DiGraph
    tumor_data = tumor_data.to_dataset_DiGraphs()

    # flatten the list with a list for each patient storing just one graph
    list_graphs = [tumor_data[i][0] for i in range(len(tumor_data))]

    # convert each DiGraph in the dataset into an undirected Graph object (required by oncotree2vec)
    list_graphs = [dir_graph.to_undirected() for dir_graph in list_graphs]

    # convert each node label, which is a list of strings, into the integer id of the single string (required by oncotree2vec, which does not work with mutation lists as node labels)
    for i, graph in enumerate(list_graphs):
        for node in graph.nodes:
            graph.nodes[node]['Label'] = labels_mapping[graph.nodes[node]['labels'][0]]    # in the case of this dataset, just one mutation per node appears as label
            del graph.nodes[node]['labels']                                                # we also change the node attribute name from 'labels' to 'Label', as required by oncotree2vec

    # create the directory to store the converted dataset
    os.makedirs(path, exist_ok=True)

    # convert each undirected graph into a .gexf file
    for i, graph in enumerate(list_graphs):
        nx.write_gexf(graph, os.path.join(path, f'patient_{i}.gexf'))

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Pre-process the input dataset and splits it into training and test sets')
    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--dataset', type=str, required=True, help='Path to the .npy file in format suitable for CloMu containing the dataset with the phylogenetic trees to be further split into training' \
    ' and test sets and used for the experiments. The "data/cancer_progression" directory contains the two datasets used for the experiments reported in the paper: "breastCancer.npy" and "AML.npy"')

    # optional arguments
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--gene_level_analysis', action='store_true', help='If set, collapse mutations at gene level')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of patients in the input dataset to be included in the training set. The other are included in the test set')
    parser.add_argument('-o', '--datasets_path', type=str, default=None, help='Path to the directory where to save all computed datasets')
    parser.add_argument('--max_tree_length', type=int, default=9, help='Maximum length in terms of edges of a phylogenetic tree to be considered')

    return parser.parse_args()    

if __name__ == '__main__':

    # parse the command line arguments
    args = parse_args()

    # paths where to save the training set, test set and a single .npy file with the test set appended at the end of the training set
    if args.datasets_path is None:
        args.datasets_path = os.path.join('..', 'results', 'cancer_progression_modeling', os.path.basename(args.dataset)[:-len('.npy')], f'random_seed_{args.random_seed}', 'pre_processed_data')
    train_set_path = os.path.join(args.datasets_path, 'train_set.npy')
    test_set_path = os.path.join(args.datasets_path, 'test_set.npy')
    train_test_path = os.path.join(args.datasets_path, 'train_test.npy')
    os.makedirs(os.path.dirname(train_set_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_set_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_test_path), exist_ok=True)

    # pre-process the input dataset and split it into training and test set
    train_set_np, test_set_np, train_data, test_data = pre_processing(
        args.dataset,
        args.max_tree_length,
        args.train_ratio,
        args.random_seed,
        train_set_path=train_set_path,
        test_set_path=test_set_path,
        train_test_path=train_test_path,
        gene_level_analysis=args.gene_level_analysis
    )

    # convert the input training set into a dataset that can be processed by RECAP and save it
    convert_to_RECAP(train_set_np, os.path.join(os.path.dirname(train_set_path), os.path.basename(train_set_path)[:-len('.npy')] + '_RECAP.txt'))

    # convert the input test set into a dataset that can be processed by oncotree2vec and save it
    convert_to_oncotree2vec(train_data, os.path.join(os.path.dirname(train_set_path), 'dataset_oncotree2vec'))