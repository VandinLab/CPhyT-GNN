import os

# set environment variable to limit the number of threads for scikit-learn
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from functools import partial
import argparse
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
import seaborn as sns
import matplotlib.pyplot as plt
from tumor_model import TumorNode
from tumor_model import TumorGraph
from tumor_model import GraphDistances
from tumor_model import TumorDataset
from tumor_model import TorchTumorDataset
from tumor_model import TumorGraphGNN
from tumor_model import TrainerTumorModel as Trainer
import utils as Utils

def base_linear_tree(n_nodes):
    """
    Generates a linear phylogenetic tree with a root node and n_nodes other nodes, each labelled by a different mutation.

    Parameters:
    - n_nodes: number of nodes in the tree (root excluded).

    Returns:
    - tree: TumorGraph object representing the linear tree.
    """

    # create the root node, with id 0 and no mutation
    root = TumorNode(0, ['root'])

    # initialize the lists of nodes and edges
    nodes = [root]
    edges = []

    # create the nodes and edges of the tree
    for i in range(1, n_nodes + 1):
        nodes.append(TumorNode(i, [str(i)]))
        edges.append((i - 1, i))

    # create the tree
    tree = TumorGraph(nodes, edges)

    return tree

def base_branching_tree(n_nodes):
    """
    Generates a phylogenetic tree with a root node and n_nodes children of it, each labelled by a different mutation.

    Parameters:
    - n_nodes: number of nodes in the tree (root excluded).

    Returns:
    - tree: TumorGraph object representing the branching tree.
    """

    # create the root node, with id 0 and no mutation
    root = TumorNode(0, ['root'])

    # initialize the lists of nodes and edges
    nodes = [root]
    edges = []

    # create the nodes and edges of the tree
    for i in range(1, n_nodes + 1):
        nodes.append(TumorNode(i, [str(i)]))
        edges.append((0, i))

    # create the tree
    tree = TumorGraph(nodes, edges)

    return tree
    
def base_binary_tree(n_nodes):
    """
    Generates a complete binary phylogenetic tree with the root node and other n_nodes nodes labelled by a different mutation.

    Parameters:
    - n_nodes: number of nodes in the tree (root excluded).

    Returns:
    - tree: TumorGraph object representing the complete tree.
    """

    # create the root node, with id 0 and no mutation
    root = TumorNode(0, ['root'])

    # initialize the lists of nodes and edges
    nodes = [root]
    edges = []

    # create the list of nodes and edges of the tree
    for i in range(1, n_nodes + 1):
        nodes.append(TumorNode(i, [str(i)]))
        parent = (i - 1) // 2
        edges.append((parent, i))

    # create the tree
    tree = TumorGraph(nodes, edges)

    return tree

def descendant_ids(n_nodes, id):
    """
    Finds the ids of all nodes in a complete binary phylogenetic tree with n_nodes nodes (root excluded) that are descendants of the node with the input id.

    Parameters:
    - n_nodes: number of nodes in the tree (root excluded).
    - id: id of the node in the tree.

    Returns:
    - eligible_ids: list with the ids of the nodes that are descendants of the node with id id.
    """

    # initialize the list of eligible ids
    eligible_ids = []
    
    # else, remove the descendants of the node with the input id
    explore_queue = [id]
    while len(explore_queue) > 0:
        curr_id = explore_queue.pop(0)
        eligible_ids.append(curr_id)
        if curr_id * 2 + 1 <= n_nodes:
            explore_queue.append(curr_id * 2 + 1)
            if curr_id * 2 + 2 <= n_nodes:
                explore_queue.append(curr_id * 2 + 2)
    
    # remove the input id
    eligible_ids.remove(id)
    
    return eligible_ids

def ancestor_ids(id):
    """
    Finds the ids of all nodes in a complete binary phylogenetic tree that are ancestors of the node with the input id.

    Parameters:
    - id: id of the node in the tree.

    Returns:
    - eligible_ids: list with the ids of the nodes that are ancestors of the node with id id.
    """

    # initialize the list of eligible ids
    eligible_ids = []
    
    # remove the ancestors of the node with the input id
    curr_id = id
    while True:
        curr_id = (curr_id - 1) // 2
        if curr_id != 0:
            eligible_ids.append(curr_id)
        else:
            break
    
    return eligible_ids

def exclusive_ids(n_nodes, id):
    """
    Finds the ids of all nodes in a complete binary phylogenetic tree with n_nodes nodes (root excluded) that do not have ancestry relations with the node with the input id.
    That is, the ids of the nodes that are not in the path from the root to the node with the input id nor descendants of it.

    Parameters:
    - n_nodes: number of nodes in the tree (root excluded).
    - id: id of the node in the tree.

    Returns:
    - eligible_ids: list with the ids of the nodes that do not have ancestry relations with the node with id id.
    """

    # initialize the list of eligible ids
    eligible_ids = [i for i in range(1, n_nodes + 1)]

    # remove the input id
    eligible_ids.remove(id)
    
    # remove the ancestors of the node with the input id
    ancestors = ancestor_ids(id)
    for ancestor in ancestors:
        eligible_ids.remove(ancestor)
    
    # else, remove the descendants of the node with the input id
    descendants = descendant_ids(n_nodes, id)
    for descendant in descendants:
        eligible_ids.remove(descendant)
    
    return eligible_ids

def random_topological_operation(tree, mutations, random_generator):
    """
    Performs a random topological operation on the input phylogenetic tree.
    In particular, the operation is one of the folowing, picked uniformly at random:
    - remove a node, except the root;
    - add a node to one of those already in the input tree, chosen uniformly at random, with mutation label chosen uniformly at random among those in the input mutation set;
    - add an empty node to a node chosen uniformly at random among those in the input tree.

    Parameters:
    - tree: TumorGraph object representing the input phylogenetic tree.
    - mutations: list of mutations to choose from.
    - random_generator: random generator for reproducibility.

    Returns:
    - mod_tree: TumorGraph object representing the modified phylogenetic tree.
    """

    # list of possible random operations
    operations = ['remove', 'add', 'add_empty']

    # choose a random operation
    operation = random_generator.choice(operations)

    # initialize the modified tree as a copy of the input tree
    mod_tree = copy.deepcopy(tree)

    # remove a node, if the chosen operation is 'remove'
    if operation == 'remove':

        # choose a random node, except for the root
        node_ids = mod_tree.get_node_ids()
        node_ids.remove(0)
        id_to_remove = random_generator.choice(node_ids)

        # create an edge for each child of the chosen node directed from the parent of the chosen node to the child
        edges = mod_tree.edges
        parent_id = None
        for edge in edges:
            if edge[1] == id_to_remove:
                parent_id = edge[0]
                break
        for edge in edges:
            if edge[0] == id_to_remove:
                mod_tree.add_edge((parent_id, edge[1]))

        # remove the chosen node from the tree and all the edges in which it appears
        mod_tree.remove_node(id_to_remove)
    
    # add a node, if the chosen operation is 'add'
    elif operation == 'add':

        # create a node with a new id and a random mutation label and add it to the tree
        node_ids = mod_tree.get_node_ids()
        id_to_add = max(node_ids) + 1
        mutation_label = random_generator.choice(mutations)
        mod_tree.add_node(TumorNode(id_to_add, [mutation_label]))

        # choose a random parent node to add the new node to and add the edge to the tree
        id_parent = random_generator.choice(node_ids)
        mod_tree.add_edge((id_parent, id_to_add))

    # add an empty node, if the chosen operation is 'add_empty'
    elif operation == 'add_empty':

        # create a node with a new id and no mutation label and add it to the tree
        node_ids = mod_tree.get_node_ids()
        id_to_add = max(node_ids) + 1
        mutation_label = random_generator.choice(mutations)
        mod_tree.add_node(TumorNode(id_to_add, ['empty']))

        # choose a random parent node to add the new node to and add the edge to the tree
        id_parent = random_generator.choice(node_ids)
        mod_tree.add_edge((id_parent, id_to_add))
    
    return mod_tree

def random_mutation_operation(tree, mutations, random_generator):
    """
    Performs a random mutation operation on the input phylogenetic tree.
    In particular, the operation is one of the folowing, picked uniformly at random:
    - remove a mutation from a node, except the root;
    - choose uniformly at random a node among those in the input tree and add a mutation chosen uniformly at random among those provided as input, but not already present in the node;
    - choose uniformly at random a node among those in the input tree and substitute a mutation labelling it with another one, chosen uniformly at random among those provided as input, but not already present in the node;
    - swap the mutations labelling two nodes, chosen uniformly at random among those in the input tree.

    Parameters:
    - tree: TumorGraph object representing the input phylogenetic tree.
    - mutations: list of mutations to choose from.
    - random_generator: random generator for reproducibility.

    Returns:
    - mod_tree: TumorGraph object representing the modified phylogenetic tree.
    """

    # list of possible random operations
    operations = ['remove', 'add', 'substitute', 'swap']

    # choose a random operation
    operation = random_generator.choice(operations)

    # initialize the modified tree as a copy of the input tree
    mod_tree = copy.deepcopy(tree)

    # remove a mutation chosen uniformly at random among those labelling a node chosen uniformly at random among those in the tree, if the chosen operation is 'remove'
    if operation == 'remove':

        # choose a random node, except for the root
        node_ids = mod_tree.get_node_ids()
        node_ids.remove(0)
        chosen_id = random_generator.choice(node_ids)
        chosen_node = [mod_tree.nodes[i] for i in range(mod_tree.n_nodes()) if mod_tree.nodes[i].id == chosen_id][0]

        # choose a random mutation from the list of mutations labelling the chosen node and remove it
        chosen_node.labels.remove(random_generator.choice(chosen_node.labels))
        if len(chosen_node.labels) == 0:
            chosen_node.labels.append('empty')

        # replace the node in the tree with the chosen id with the modified one that has the chosen mutation removed
        mod_tree.add_node(chosen_node)
    
    # add a mutation chosen uniformly at random among those provided as input to a node chosen uniformly at random among those in the tree, if the chosen operation is 'add'
    elif operation == 'add':

        # choose a random node, except for the root
        node_ids = mod_tree.get_node_ids()
        node_ids.remove(0)
        chosen_id = random_generator.choice(node_ids)
        chosen_node = [mod_tree.nodes[i] for i in range(mod_tree.n_nodes()) if mod_tree.nodes[i].id == chosen_id][0]

        # choose a random mutation from the list of mutations provided as input, but not already present in the node, and add it to the list of mutations labelling the chosen node
        eligible_mutations = list(set(mutations) - set(chosen_node.labels))
        chosen_label = random_generator.choice(eligible_mutations)
        
        # add the chosen mutation to the list of mutations labelling the chosen node
        chosen_node.labels.append(chosen_label)

        # if the node was previously empty, then remove the empty label since it is no more empty
        if 'empty' in chosen_node.labels:
            chosen_node.labels.remove('empty')

        # replace the node in the tree with the chosen id with the modified one that has the chosen mutation added
        mod_tree.add_node(chosen_node)

    # choose uniformly at random a node among those in the input tree and substitute a mutation labelling it with another one, chosen uniformly at random among those provided as input, but not already present in the node, if the chosen operation is 'substitute'
    elif operation == 'substitute':

        # choose a random node, except for the root
        node_ids = mod_tree.get_node_ids()
        node_ids.remove(0)
        chosen_id = random_generator.choice(node_ids)
        chosen_node = [mod_tree.nodes[i] for i in range(mod_tree.n_nodes()) if mod_tree.nodes[i].id == chosen_id][0]

        # choose a random mutation from the list of mutations provided as input, but not already present in the node, and add it to the list of mutations labelling the chosen node
        eligible_mutations = list(set(mutations) - set(chosen_node.labels))
        chosen_new_label = random_generator.choice(eligible_mutations)

        # choose a random mutation from the list of mutations labelling the chosen node
        chosen_old_label = random_generator.choice(chosen_node.labels)

        # substitute the chosen mutation with the new one
        chosen_node.labels.remove(chosen_old_label)
        chosen_node.labels.append(chosen_new_label)

        # replace the node in the tree with the chosen id with the modified one that has the chosen mutation added
        mod_tree.add_node(chosen_node)

    # swap the mutations labelling two nodes, chosen uniformly at random among those in the input tree, if the chosen operation is 'swap'
    elif operation == 'swap':

        # choose two random nodes, except for the root
        node_ids = mod_tree.get_node_ids()
        node_ids.remove(0)
        chosen_ids = random_generator.choice(node_ids, size=2, replace=False)
        chosen_nodes = [mod_tree.nodes[i] for i in range(mod_tree.n_nodes()) if mod_tree.nodes[i].id in chosen_ids]

        # swap the labels of the two chosen nodes
        curr_labels = chosen_nodes[0].labels
        chosen_nodes[0].labels = chosen_nodes[1].labels
        chosen_nodes[1].labels = curr_labels

        # replace the original chosen nodes in the tree with the modified ones that have the labels swapped
        mod_tree.add_node(chosen_nodes[0])
        mod_tree.add_node(chosen_nodes[1])

    return mod_tree

def cluster_embeddings(embeddings, k):
    """
    Computes a k-means clustering of the input embeddings for the value of k provided as input.

    Parameters:
    - embeddings: array with the input embeddings to be clustered.
    - k: number of clusters.

    Returns:
    - cluster_labels: array with cluster labels for embeddings.
    """

    # scale the embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # cluster the embeddings
    kmeans = KMeans(n_clusters=k, init='k-means++', algorithm='lloyd', n_init=1000, copy_x=True, max_iter=1000)
    cluster_labels = kmeans.fit_predict(embeddings) 
    
    return cluster_labels

def baseline_encoding(train_labels_mapping, graphs):
    """
    Encode each TumorGraph in the input list as a binary vector indicating the presence of mutations, using the mutations that appear in the training set.

    Parameters:
    - train_labels_mapping: dictionary that maps the node labels to consecutive integers starting from 0.
    - graphs: list of TumorGraph objects to encode.
    
    Returns:
    - features: numpy array with the binary encoding of the input graphs. It is aligned with the input graphs, meaning that the i-th row of the array corresponds to the i-th graph in the input list.
    """
    
    # compute the feature matrix for the input graphs
    features = np.zeros((len(graphs), len(train_labels_mapping) - 1), dtype=np.int64)
    for i, graph in enumerate(graphs):
        for node in graph.nodes:
            for label in node.labels:
                if label != "empty":
                    features[i, train_labels_mapping[label]] = 1
    
    return features

def plot_true_clusters(distances, cluster_labels, save_path, title=''):
    """
    Generates a heatmap of the distances between all pairs of phylogenetic trees in a dataset and saves it.

    Parameters:
    - distances: numpy array with the distances between all pairs of phylogenetic trees in the dataset.
    - cluster_labels: list with the cluster label of each tree in the dataset. It must be aligned with the distances array,
                      meaning that the i-th element of the list corresponds to the i-th row and column of the distances array.
    - save_path: path to the directory where to save the heatmap.
    - title: title of the heatmap.
    """

    # create a dataframe with the distances
    df = pd.DataFrame(distances, index=cluster_labels, columns=cluster_labels)

    # create a heatmap with the distances
    plt.close()
    ax = sns.heatmap(df, cmap='Reds', annot=False, square=True, cbar_kws={'label': 'Distance'})
    
    # add horizontal and vertical lines delimiting the clusters
    unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    boundaries = np.cumsum(cluster_sizes)
    for b in boundaries[:-1]:  # skip the last boundary since it's the edge
        ax.axhline(b, color='blue', lw=1)
        ax.axvline(b, color='blue', lw=1)
    
    # set ticks only at the center of each cluster
    tick_positions = (np.cumsum(cluster_sizes) - cluster_sizes / 2)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(unique_labels, rotation=0)
    ax.set_yticklabels(unique_labels)

    # remove the tick marks
    ax.tick_params(axis='both', which='both', length=0)

    # set title and save the figure
    ax.set_title(title)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cluster')
    plt.savefig(save_path, dpi=350)
    plt.close()

def base_simulation(generated_trees, node_encoding_type, curr_sim_save, device, include_baseline=True):
    """
    Core function to run every simulation.
    It contains everything necessary to train a GNN-based model on a generated datset, compute the embeddings, cluster them, compute baseline embeddings and cluster them.

    Parameters:
    - generated_trees: dictionary containing the generated phylogenetic trees for each cluster.
                       The keys are the identifiers of the clusters and the values are dictionaries of patients for the corresponding cluster.
                       Each patient is identified by a unique id and has a list of trees with just the corresponding generated tree as value.
    - node_encoding_type: type of node encoding to use: 'mutation' or 'clone'.
    - curr_sim_save: path to the directory where to save the results of the current simulation.
    - device: device to use for training and evaluation: 'cpu', 'cuda' or 'mps'.
    - include_baseline: if True, compute the baseline feature vectors and cluster them, otherwise do not consider the baseline.

    Returns:
    - rand_index_baseline: Rand index of the clustering obtained with the baseline feature vectors. It is None if include_baseline is False.
    - rand_index_GNN: Rand index of the clustering obtained with the GNN-based embeddings.
    """

    # set the hyper parameters for the GNN
    hyper_parameters = {
        'h_1': 64,
        'h_2': 64,
        'embedding_dim': 32,
        'dropout_prob_1': 0.0,
        'dropout_prob_2': 0.0,
        'batch_normalization': True,
        'loss_fn': 'MAE_loss',
        'optimizer': 'Adam',
        'weight_decay': 0.0,
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 50,
        'verbose': True,
        'save_plot': True,
    }

    # sort the ids of the generated trees
    sorted_keys = sorted(generated_trees.keys())

    # extract the true cluster labels for the generated trees
    true_labels = []
    for i, key in enumerate(sorted_keys):
        true_labels.append(key.split('_')[0])
    true_labels = np.array(true_labels)

    # create an array with true cluster labels for all the generated trees converted to integers
    labels_mapping = {label: i for i, label in enumerate(np.unique(true_labels))}
    true_labels_int = np.empty(len(sorted_keys), dtype=np.int64)
    for i, key in enumerate(sorted_keys):
        curr_cluster = key.split('_')[0]
        true_labels_int[i] = labels_mapping[curr_cluster]

    # create a TumorDataset with the generated trees
    dataset = [generated_trees[sorted_keys[i]] for i in range(len(sorted_keys))]
    train_data = TumorDataset(dataset)

    # convert the dataset into a TorchTumorDataset object
    train_torch_data = TorchTumorDataset(train_data, node_encoding_type=node_encoding_type)

    # compute the tensor with the distances between all pairs of graphs in the training dataset
    train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

    # create a heatmap with the distances
    plot_true_clusters(train_distances, true_labels, os.path.join(curr_sim_save, 'true_distances.jpg'), title='True Distances')

    # print information
    print('Training our GNN-based model to compute embeddings...')

    # create a TumorGraphGNN instance with input size based on the labels in the training set
    model = TumorGraphGNN(
        n_node_labels=len(train_data.node_labels()),
        h_1=hyper_parameters['h_1'],
        h_2=hyper_parameters['h_2'],
        embedding_dim=hyper_parameters['embedding_dim'],
        dropout_prob_1=hyper_parameters['dropout_prob_1'],
        dropout_prob_2=hyper_parameters['dropout_prob_2'],
        batch_normalization=hyper_parameters['batch_normalization'],
        device=device
    )

    # set the path where to save the training plot
    plot_save = None
    if hyper_parameters['save_plot']:
        plot_save = os.path.join(curr_sim_save, 'GNN_training_loss.jpg')

    # train the model on the training set with the input parameters
    Trainer.train(
        model=model,
        train_data=train_torch_data,
        train_graph_distances=train_distances,
        loss_fn=Utils.select_loss(hyper_parameters['loss_fn']),
        batch_size=hyper_parameters['batch_size'],
        optimizer=Utils.select_optimizer(hyper_parameters['optimizer']),
        weight_decay=hyper_parameters['weight_decay'],
        plot_save=plot_save,
        verbose=hyper_parameters['verbose'],
        epochs=hyper_parameters['epochs'],
        lr=hyper_parameters['learning_rate'],
        device=device,
        save_model=os.path.join(curr_sim_save, 'GNN_weights.pth'),
    )

    # print information
    print('Computing the GNN embeddings...')

    # create a dataloader storing the phylogenetic trees
    train_dataloader = Trainer.get_dataloader(train_torch_data, batch_size=hyper_parameters['batch_size'], shuffle=False)

    # compute the embeddings for the phylogenetic trees in the training set
    embeddings = Trainer.get_embeddings(model, train_dataloader, device=device)
    embeddings = embeddings.cpu().detach().numpy()

    # print information
    print('Clustering the GNN embeddings...')

    # compute the cluster labels for the computed embeddings
    predicted_labels = cluster_embeddings(embeddings, k=len(np.unique(true_labels)))

    # print information
    print('Evaluating the GNN-based clustering...')

    # compare the true labels with the predicted ones
    rand_index_GNN = rand_score(true_labels_int, predicted_labels)

    # consider the baseline feature vectors only if include_baseline is True
    if not include_baseline:
        return None, rand_index_GNN

    # print information
    print('Computing the baseline feature vectors...')

    # represent each generated phylogenetic tree as a vector indicating the presence of the mutations
    list_trees = [tree for patient in train_data.dataset for tree in patient]
    baseline_features = baseline_encoding(TorchTumorDataset.map_node_labels(train_data.node_labels()), list_trees)

    # print information
    print('Clustering the baseline feature vectors...')

    # compute the cluster labels for the computed embeddings
    predicted_labels = cluster_embeddings(baseline_features, k=len(np.unique(true_labels)))

    # print information
    print('Evaluating the baseline clustering...')

    # compare the true labels with the predicted ones
    rand_index_baseline = rand_score(true_labels_int, predicted_labels)

    return rand_index_baseline, rand_index_GNN

def generate_data_simulation_I(save_path, n_nodes, n_samples, n_random_ops, random_generator, no_binary=False):
    """
    Generates data for simulation I.

    Parameters:
    - save_path: path to the directory where to save the generated data.
    - n_nodes: number of nodes (root excluded) in each base tree of each cluster.
    - n_samples: number of samples to generate for each cluster.
    - n_random_ops: number of random topological operations to perform on each base tree.
    - random_generator: random generator for reproducibility.
    - no_binary: if True, do not generate trees with binary topology.

    Returns:
    - generated_trees: dictionary containing the generated phylogenetic trees for each cluster.
                       The keys are the identifiers of the clusters and the values are dictionaries of patients for the corresponding cluster.
                       Each patient is identified by a unique id and has a list of trees with just the corresponding generated tree as value.
    """

    # identifiers of the clusters
    cluster_ids = ['linear', 'branching']

    # generate the base tree for each cluster
    base_trees = {
        cluster_ids[0]: base_linear_tree(n_nodes),
        cluster_ids[1]: base_branching_tree(n_nodes),
    }

    # generate the base binary tree, if required
    if not no_binary:
        cluster_ids.append('binary')
        base_trees[cluster_ids[2]] = base_binary_tree(n_nodes)

    # create the list of mutations as those in the base trees union with mutations not present
    mutation_list = [str(i) for i in range(1, 2 * n_nodes + 1)]

    # initialize the dictionary that will contain the generated phylogenetic trees
    generated_trees = {}         # each tree is saved into a dictionary with an id for the tree as key and a list of trees with just the corresponding tree as value

    # generate the phylogenetic trees for each cluster by performing random operations on the base trees
    for cluster_id in cluster_ids:
        for j in range(n_samples):

            # get the base tree for the current cluster
            curr_tree = base_trees[cluster_id]
            
            # perform the specified number of random topological operations on the base tree
            for k in range(n_random_ops):
                curr_tree = random_topological_operation(curr_tree, mutation_list, random_generator)

            # add the generated tree to the dictionary of patients, where each patient is identified by a unique id and has a list of trees with just the corresponding generated tree
            generated_trees[f'{cluster_id}_{j}'] = [curr_tree]

    # save the generated trees to the specified path
    Utils.save_dataset_txt_survival(generated_trees, os.path.join(save_path, 'generated_trees.txt'))

    return generated_trees

def generate_data_simulation_II(save_path, n_nodes, n_samples, random_generator, no_binary=False):
    """
    Generates data for simulation II.

    Parameters:
    - save_path: path to the directory where to save the generated data.
    - n_nodes: number of nodes (root excluded) in each base tree of each cluster.
    - n_samples: number of samples to generate for each cluster.
    - random_generator: random generator for reproducibility.
    - no_binary: if True, do not generate trees with binary topology.

    Returns:
    - generated_trees: dictionary containing the generated phylogenetic trees for each cluster.
                       The keys are the identifiers of the clusters and the values are dictionaries of patients for the corresponding cluster.
                       Each patient is identified by a unique id and has a list of trees with just the corresponding generated tree as value.
    """

    # identifiers of the clusters
    cluster_ids = ['linear', 'branching']
    if not no_binary:
        cluster_ids.append('binary')

    # dictionary to store the base trees for each cluster
    base_trees = {}

    # generate the base linear tree
    root = TumorNode(0, ['root'])
    nodes = [root]
    edges = []
    for i in range(1, n_nodes + 1):
        nodes.append(TumorNode(i, ['empty']))
        edges.append((i - 1, i))
    base_trees[cluster_ids[0]] = TumorGraph(nodes, edges)

    # generate the base branching tree
    root = TumorNode(0, ['root'])
    nodes = [root]
    edges = []
    for i in range(1, n_nodes + 1):
        nodes.append(TumorNode(i, ['empty']))
        edges.append((0, i))
    base_trees[cluster_ids[1]] = TumorGraph(nodes, edges)

    # generate the base binary tree, if requested
    if not no_binary:
        root = TumorNode(0, ['root'])
        nodes = [root]
        edges = []
        for i in range(1, n_nodes + 1):
            nodes.append(TumorNode(i, ['empty']))
            parent = (i - 1) // 2
            edges.append((parent, i))
        base_trees[cluster_ids[2]] = TumorGraph(nodes, edges)

    # create a list with n_nodes * len(base_trees) mutations that can be used to label the nodes in the trees, plus the empty label
    mutation_list = [str(i) for i in range(1, len(base_trees) * n_nodes + 1)] + ['empty']

    # initialize the dictionary that will contain the generated phylogenetic trees
    generated_trees = {}         # each tree is saved into a dictionary with an id for the tree as key and a list of trees with just the corresponding tree as value

    # generate the phylogenetic trees for each cluster by assigning a random label to each node
    for cluster_id in cluster_ids:
        for j in range(n_samples):
            base_tree = base_trees[cluster_id]
            curr_tree = copy.deepcopy(base_tree)
            for node_idx in range(len(curr_tree.nodes)):
                if curr_tree.nodes[node_idx].id != 0:
                    curr_tree.nodes[node_idx].labels = [random_generator.choice(mutation_list)]
            generated_trees[f'{cluster_id}_{j}'] = [curr_tree]

    # save the generated trees to the specified path
    Utils.save_dataset_txt_survival(generated_trees, os.path.join(save_path, 'generated_trees.txt'))

    return generated_trees

def generate_data_simulation_III(save_path, n_nodes):
    """
    Generates data for simulation III.

    Parameters:
    - save_path: path to the directory where to save the generated data.
    - n_nodes: maximum number of nodes (root excluded) in each tree.

    Returns:
    - generated_trees: dictionary containing the generated phylogenetic trees for each cluster.
                       The keys are the identifiers of the clusters and the values are dictionaries of patients for the corresponding cluster.
                       Each patient is identified by a unique id and has a list of trees with just the corresponding generated tree as value.
    """

    # initialize the dictionary that will contain the generated phylogenetic trees
    generated_trees = {}         # each tree is saved into a dictionary with an id for the tree as key and a list of trees with just the corresponding tree as value

    # generate empty phylogenetic trees with linear topology of different sizes
    for i in range(2, n_nodes + 1):

        # create the root node, with id 0 and no mutation
        root = TumorNode(0, ['root'])

        # initialize the lists of nodes and edges
        nodes = [root]
        edges = []

        # create the nodes and edges of the tree
        for j in range(1, i + 1):
            nodes.append(TumorNode(j, ['empty']))   # each node is empty
            edges.append((j - 1, j))

        # create the tree
        tree = TumorGraph(nodes, edges)

        # add the tree to the dictionary of patients
        generated_trees[f'linear_{j}'] = [tree]
    
    # generate empty phylogenetic trees with branching topology of different sizes
    for i in range(2, n_nodes + 1):

        # create the root node, with id 0 and no mutation
        root = TumorNode(0, ['root'])

        # initialize the lists of nodes and edges
        nodes = [root]
        edges = []

        # create the nodes and edges of the tree
        for j in range(1, i + 1):
            nodes.append(TumorNode(j, ['empty']))   # each node is empty
            edges.append((0, j))

        # create the tree
        tree = TumorGraph(nodes, edges)

        # add the tree to the dictionary of patients
        generated_trees[f'branching_{j}'] = [tree]

    # save the generated trees to the specified path
    Utils.save_dataset_txt_survival(generated_trees, os.path.join(save_path, 'generated_trees.txt'))

    return generated_trees

def generate_data_simulation_IV(save_path, n_nodes, no_exclusives=False, no_descendants=False):
    """
    Generates data for simulation IV.

    Parameters:
    - save_path: path to the directory where to save the generated data.
    - n_nodes: maximum number of nodes (root excluded) in each tree.
    - no_exclusives: if True, do not generate trees with exclusivity relations.
    - no_descendants: if True, do not generate trees with descendancy relations.

    Returns:
    - generated_trees: dictionary containing the generated phylogenetic trees for each cluster.
                       The keys are the identifiers of the clusters and the values are dictionaries of patients for the corresponding cluster.
                       Each patient is identified by a unique id and has a list of trees with just the corresponding generated tree as value.
    """

    # initialize the dictionary that will contain the generated phylogenetic trees
    generated_trees = {}         # each tree is saved into a dictionary with an id for the tree as key and a list of trees with just the corresponding tree as value

    # ids of the clusters
    cluster_ids = ['exclusivity', 'ancestor', 'descendant']

    # define two mutations that will be the only one present in the trees
    mutations = ['1', '2']

    # id of the last non-leaf node
    last_non_leaf_id = (n_nodes - 1) // 2

    # generate all possible complete binary phylogenetic trees with n_nodes where mutations[0] is inserted in a node with different id i
    for i in range(1, last_non_leaf_id + 1):

        # create a binary tree with all nodes, except for the root, that are empty
        root = TumorNode(0, ['root'])
        nodes = [root]
        edges = []
        for j in range(1, n_nodes + 1):
            nodes.append(TumorNode(j, ['empty']))
            parent = (j - 1) // 2
            edges.append((parent, j))
        tree = TumorGraph(nodes, edges)

        # insert mutations[0] to the node with id i
        tree.add_node(TumorNode(i, [mutations[0]]))

        # generate all possible phylogenetic trees where mutations[1] is in a node that is not in the path from the root to the node with id i nor a descendant of it and is not a leaf
        if not no_exclusives:
            eligible_ids = exclusive_ids(n_nodes, i)
            for eligible_id in eligible_ids:
                if eligible_id * 2 + 1 <= n_nodes:   # do not consider leaf nodes
                    curr_tree = copy.deepcopy(tree)
                    curr_tree.add_node(TumorNode(eligible_id, [mutations[1]]))
                    generated_trees[f'{cluster_ids[0]}_{i}_{eligible_id}'] = [curr_tree]
        
        # generate all possible complete binary phylogenetic trees where mutations[1] is in a node that is descendant of the node with id i and is not a leaf
        if not no_descendants:
            eligible_ids = descendant_ids(n_nodes, i)
            for eligible_id in eligible_ids:
                if eligible_id * 2 + 1 <= n_nodes:   # do not consider leaf nodes
                    curr_tree = copy.deepcopy(tree)
                    curr_tree.add_node(TumorNode(eligible_id, [mutations[1]]))
                    generated_trees[f'{cluster_ids[1]}_{i}_{eligible_id}'] = [curr_tree]

        # generate all possible complete binary phylogenetic trees where mutations[1] is in a node that is an ancestor of the node with id i and is not a leaf
        eligible_ids = ancestor_ids(i)
        for eligible_id in eligible_ids:
            if eligible_id * 2 + 1 <= n_nodes:   # do not consider leaf nodes
                curr_tree = copy.deepcopy(tree)
                curr_tree.add_node(TumorNode(eligible_id, [mutations[1]]))
                generated_trees[f'{cluster_ids[2]}_{i}_{eligible_id}'] = [curr_tree]

    # save the generated trees to the specified path
    Utils.save_dataset_txt_survival(generated_trees, os.path.join(save_path, 'generated_trees.txt'))

    return generated_trees

def generate_data_simulation_V(save_path, n_nodes, no_descendants=False, no_exclusives=False):
    """
    Generates data for simulation V.

    Parameters:
    - save_path: path to the directory where to save the generated data.
    - n_nodes: maximum number of nodes (root excluded) in each tree.
    - no_descendants: if True, do not generate trees with descendants.
    - no_exclusives: if True, do not generate trees with exclusives.

    Returns:
    - generated_trees: dictionary containing the generated phylogenetic trees for each cluster.
                       The keys are the identifiers of the clusters and the values are dictionaries of patients for the corresponding cluster.
                       Each patient is identified by a unique id and has a list of trees with just the corresponding generated tree as value.
    """

    # initialize the dictionary that will contain the generated phylogenetic trees
    generated_trees = {}         # each tree is saved into a dictionary with an id for the tree as key and a list of trees with just the corresponding tree as value

    # ids of the clusters
    cluster_ids = ['exclusivity', 'ancestor', 'descendant']

    # define two mutations that will be the only one present in the trees
    mutations = ['1', '2']

    # create a binary tree with all nodes, except for the root, that are empty
    root = TumorNode(0, ['root'])
    nodes = [root]
    edges = []
    for j in range(1, n_nodes + 1):
        nodes.append(TumorNode(j, ['empty']))
        parent = (j - 1) // 2
        edges.append((parent, j))
    tree = TumorGraph(nodes, edges)

    # generate all possible complete binary phylogenetic trees with n_nodes where mutations[0] and mutations[1] are inserted in all possible nodes except the root
    for i in range(1, n_nodes + 1):

        # create a tree with two additional nodes childen of the node with id i, one with mutation[0] and the other with mutation[1]
        if not no_exclusives:
            curr_tree = copy.deepcopy(tree)
            curr_tree.add_node(TumorNode(n_nodes + 1, [mutations[0]]))
            curr_tree.add_node(TumorNode(n_nodes + 2, [mutations[1]]))
            curr_tree.add_edge((i, n_nodes + 1))
            curr_tree.add_edge((i, n_nodes + 2))
            generated_trees[f'{cluster_ids[0]}_{i}'] = [curr_tree]

        # create a tree with two additional nodes: one that is child of the node with id i and carries mutations[0] and the other that is child of the one with mutations[0] and carries mutations[1]
        curr_tree = copy.deepcopy(tree)
        curr_tree.add_node(TumorNode(n_nodes + 1, [mutations[0]]))
        curr_tree.add_node(TumorNode(n_nodes + 2, [mutations[1]]))
        curr_tree.add_edge((i, n_nodes + 1))
        curr_tree.add_edge((n_nodes + 1, n_nodes + 2))
        generated_trees[f'{cluster_ids[1]}_{i}'] = [curr_tree]

        # create a tree with two additional nodes: one that is child of the node with id i and carries mutations[1] and the other that is child of the one with mutations[1] and carries mutations[0]
        if not no_descendants:
            curr_tree = copy.deepcopy(tree)
            curr_tree.add_node(TumorNode(n_nodes + 1, [mutations[0]]))
            curr_tree.add_node(TumorNode(n_nodes + 2, [mutations[1]]))
            curr_tree.add_edge((i, n_nodes + 2))
            curr_tree.add_edge((n_nodes + 2, n_nodes + 1))
            generated_trees[f'{cluster_ids[2]}_{i}'] = [curr_tree]

    # save the generated trees to the specified path
    Utils.save_dataset_txt_survival(generated_trees, os.path.join(save_path, 'generated_trees.txt'))

    return generated_trees

def simulation_I(output_dir, data_generation_fn, simulation_name, n_nodes, n_samples, n_random_ops_list, n_experiments, node_encoding_type, random_generator, device, include_baseline=True):
    """
    Runs simulation I.

    Parameters:
    - output_dir: path to the directory where to save data and results for all experiment iterations.
    - data_generation_fn: function to generate the data for the simulation.
    - simulation_name: name of the simulation.
    - n_nodes: number of nodes (root excluded) in each base tree of each cluster.
    - n_samples: number of samples to generate for each cluster.
    - n_random_ops_list: list with different number of random topological operations to perform on the base trees.
    - n_experiments: number of repetitions of the simulation for each value in n_random_ops with different random seeds controlling the random operations.
    - node_encoding_type: type of node encoding to use: 'mutation' or 'clone'.
    - random_generator: random generator to use for reproducibility.
    - device: device to use for training and evaluation: 'cpu', 'cuda' or 'mps'.
    - include_baseline: whether to include the baseline clustering based on the presence of mutations in the results.
    """

    # initialize the dataframe that will contain the results
    scores_sim = pd.DataFrame(columns=['Simulation', 'Experiment ID', 'Rand Index', 'Features', 'Random Operations', 'Base Nodes', 'Samples'])

    # path to the directory where to store the results for all experiments
    results_save = os.path.join(output_dir, f'{simulation_name}', 'results')
    os.makedirs(results_save, exist_ok=True)

    # print some information
    print(f'Values of random operations to be tested: {n_random_ops_list}')

    # iterate through the number of random operations
    for n_random_ops in n_random_ops_list:

        # print some information
        print(f'Running simulation {simulation_name} with {n_random_ops} random operations...')

        # create a directory where to store the results for the current number of random operations
        curr_random_ops_save = os.path.join(output_dir, f'{simulation_name}', f'{n_random_ops}_random_ops')

        # iterate through the number of experiments
        for exp_id in range(n_experiments):

            # print some information
            print(f'Experiment repetition {exp_id}/{n_experiments - 1}\nGenerating data...')

            # path to the directory where to store data for the current experiment
            curr_sim_save = os.path.join(curr_random_ops_save, f'repetition {exp_id}')
            os.makedirs(curr_sim_save, exist_ok=True)

            # generate the data
            generated_trees = data_generation_fn(curr_sim_save, n_nodes=n_nodes, n_samples=n_samples, n_random_ops=n_random_ops, random_generator=random_generator)

            # compute the Rand index for the baseline clustering and the GNN-based clustering
            rand_index_baseline, rand_index_GNN = base_simulation(generated_trees, node_encoding_type, curr_sim_save, device, include_baseline)

            # concatenate the dataframe with the results of the baseline clustering, if computed
            if include_baseline:
                scores_baseline = pd.DataFrame([{
                    'Experiment ID': exp_id,
                    'Simulation': f'{simulation_name}',
                    'Features': 'Baseline',
                    'Rand Index': rand_index_baseline,
                    'Random Operations': n_random_ops,
                    'Base Nodes': n_nodes,
                    'Samples': n_samples
                }])
                scores_sim = pd.concat([scores_sim, scores_baseline], ignore_index=True)

            # concatenate the dataframe with the results of the GNN-based clustering
            scores_GNN = pd.DataFrame([{
                'Experiment ID': exp_id,
                'Simulation': f'{simulation_name}',
                'Features': 'GNN',
                'Rand Index': rand_index_GNN,
                'Random Operations': n_random_ops,
                'Base Nodes': n_nodes,
                'Samples': n_samples
            }])
            scores_sim = pd.concat([scores_sim, scores_GNN], ignore_index=True)

    # print information
    print('Saving the results...')

    # save the scores
    scores_sim.to_csv(os.path.join(results_save, 'scores.csv'), index=False)

    # print information
    print(f'Simulation {simulation_name} completed successfully')

def simulation_II(output_dir, data_generation_fn, simulaton_name, n_nodes_list, n_exp, node_encoding_type, random_seed, device, include_baseline=True):
    """
    Runs simulation II.

    Parameters:
    - output_dir: path to the directory where to save data and results of the simulation.
    - data_generation_fn: function to generate the data for the simulation.
                          The function must have only the following mandatory parameters: the path where to save the generated data and the number of nodes in the trees.
    - simulaton_name: name of the simulation (e.g., 'III' or 'IX').
    - n_nodes_list: list with different number of nodes (root excluded) in each tree of each cluster.
    - n_exp: number of repetitions of the simulation for each value in n_nodes_list with different randomly generated data.
    - node_encoding_type: type of node encoding to use: 'mutation' or 'clone'.
    - random_seed: random seed for reproducibility.
    - device: device to use for training and evaluation: 'cpu', 'cuda' or 'mps'.
    - include_baseline: whether to include the baseline clustering based on the presence of mutations in the results.
    """

    # initialize the dataframe that will contain the results
    scores_sim = pd.DataFrame(columns=['Simulation', 'Experiment ID', 'Rand Index', 'Max Nodes'])

    # path to the directory where to store the results for all experiments
    results_save = os.path.join(output_dir, simulaton_name, 'results')
    os.makedirs(results_save, exist_ok=True)

    # print some information
    print(f'Tree sizes to be tested: {n_nodes_list} nodes')

    # iterate through the number of nodes in the trees
    for n_nodes in n_nodes_list:

        # print some information
        print(f'Running simulation {simulaton_name} with {n_nodes} maximum nodes per tree...')

        # iterate through the number of experiments
        for exp_id in range(n_exp):

            # create a directory where to store the results for the current experiment repetition
            curr_save = os.path.join(output_dir, simulaton_name, f'{n_nodes}_nodes', f'repetition {exp_id}')
            os.makedirs(curr_save, exist_ok=True)

            # print some information
            print(f'Experiment iteration {exp_id}/{n_exp - 1}\nGenerating data...')

            # generate the data
            generated_trees = data_generation_fn(curr_save, n_nodes=n_nodes)

            # compute the Rand index for the baseline clustering and the GNN-based clustering
            rand_index_baseline, rand_index_GNN = base_simulation(generated_trees, node_encoding_type, curr_save, device, include_baseline)

            # concatenate the results of the baseline clustering with all the results computed so far
            if include_baseline:
                scores_baseline = pd.DataFrame([{
                    'Experiment ID': exp_id,
                    'Simulation': simulaton_name,
                    'Features': 'Baseline',
                    'Rand Index': rand_index_baseline,
                    'Base Nodes': n_nodes
                }])
                scores_sim = pd.concat([scores_sim, scores_baseline], ignore_index=True)

            # save rand index and silhouette score in a dataframe
            scores_GNN = pd.DataFrame([{
                'Experiment ID': exp_id,
                'Features': 'GNN',
                'Simulation': simulaton_name,
                'Rand Index': rand_index_GNN,
                'Base Nodes': n_nodes
            }])
            scores_sim = pd.concat([scores_sim, scores_GNN], ignore_index=True)

    # print information
    print('Saving the results...')

    # save the scores
    scores_sim.to_csv(os.path.join(results_save, 'scores.csv'), index=False)

    # print information
    print(f'Simulation {simulaton_name} completed successfully')

def other_simulations(output_dir, data_generation_fn, simulaton_name, n_nodes_list, node_encoding_type, random_seed, device, include_baseline=True):
    """
    Same function to run simulations III, IV and V.

    Parameters:
    - output_dir: path to the directory where to save data and results of the simulation.
    - data_generation_fn: function to generate the data for the simulation.
                          The function must have only the following mandatory parameters: the path where to save the generated data and the number of nodes in the trees.
    - simulaton_name: name of the simulation (e.g., 'III', 'IV' or 'V').
    - n_nodes_list: list with different maximum number of nodes (root excluded) in each tree of each cluster.
    - node_encoding_type: type of node encoding to use: 'mutation' or 'clone'.
    - random_seed: random seed for reproducibility.
    - device: device to use for training and evaluation: 'cpu', 'cuda' or 'mps'.
    - include_baseline: whether to include the baseline clustering based on the presence of mutations in the results.
    """

    # initialize the dataframe that will contain the results
    scores_sim = pd.DataFrame(columns=['Simulation', 'Experiment ID', 'Rand Index', 'Max Nodes'])

    # path to the directory where to store the results for all experiments
    results_save = os.path.join(output_dir, simulaton_name, 'results')
    os.makedirs(results_save, exist_ok=True)

    # print some information
    print(f'Maximum tree sizes to be tested: {n_nodes_list} nodes')

    # iterate through the maximum number of nodes in the trees
    for n_nodes in n_nodes_list:

        # print some information
        print(f'Running simulation {simulaton_name} with {n_nodes} maximum nodes per tree...')

        # create a directory where to store the results for the current number of random operations
        curr_n_nodes_save = os.path.join(output_dir, simulaton_name, f'{n_nodes}_nodes')
        os.makedirs(curr_n_nodes_save, exist_ok=True)

        # print some information
        print('Generating data...')

        # generate the data
        generated_trees = data_generation_fn(curr_n_nodes_save, n_nodes=n_nodes)

        # compute the Rand index for the baseline clustering and the GNN-based clustering
        rand_index_baseline, rand_index_GNN = base_simulation(generated_trees, node_encoding_type, curr_n_nodes_save, device, include_baseline)

        # concatenate the results of the baseline clustering with all the results computed so far
        if include_baseline:
            scores_baseline = pd.DataFrame([{
                'Experiment ID': random_seed,
                'Simulation': simulaton_name,
                'Features': 'Baseline',
                'Rand Index': rand_index_baseline,
                'Base Nodes': n_nodes
            }])
            scores_sim = pd.concat([scores_sim, scores_baseline], ignore_index=True)

        # save rand index and silhouette score in a dataframe
        scores_GNN = pd.DataFrame([{
            'Experiment ID': random_seed,
            'Features': 'GNN',
            'Simulation': simulaton_name,
            'Rand Index': rand_index_GNN,
            'Base Nodes': n_nodes
        }])
        scores_sim = pd.concat([scores_sim, scores_GNN], ignore_index=True)

    # print information
    print('Saving the results...')

    # save the scores
    scores_sim.to_csv(os.path.join(results_save, 'scores.csv'), index=False)

    # print information
    print(f'Simulation {simulaton_name} completed successfully')

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description="Run our GNN-based model on synthetic data, performing different kinds of simulations.")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-o', '--output_dir', type=str, help='Path to the directory where to save data and results of the simulations', required=True)

    # optional arguments
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--n_nodes_I', type=int, default=8, help='Number of nodes (root excluded) in each base tree of simulation I')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate for each cluster in simulations I and II')
    parser.add_argument('--n_random_ops', type=int, nargs='+', default=list(range(1, 9)), help='List with different number of random topological operations to perform on the base trees in simulation I')
    parser.add_argument('--n_exp', type=int, default=10, help='Number of repetitions of the same simulation experiment for each value in n_random_ops with different random seeds controlling randomly generated data. \
                        Used in simulaton I and II')
    parser.add_argument('--n_nodes_list', type=int, nargs='+', default=list(range(8, 21)), help='List with different maximum number of nodes (root excluded) in each tree of each cluster in simulation II, III, IV and V')
    parser.add_argument('--no_sim_I', action='store_true', help='Do not run simulation I')
    parser.add_argument('--no_sim_II', action='store_true', help='Do not run simulation II')
    parser.add_argument('--no_sim_III', action='store_true', help='Do not run simulation III')
    parser.add_argument('--no_sim_IV', action='store_true', help='Do not run simulation IV')
    parser.add_argument('--no_sim_V', action='store_true', help='Do not run simulation V')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training and evaluation: "cpu", "cuda" or "mps"')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Type of node encoding to use: "mutation" or "clone"')
    parser.add_argument('--max_n_cores', type=int, default=1, help='Max number of CPU cores for PyTorch')
    
    return parser.parse_args()

if __name__ == '__main__':

    # parse the command line arguments
    args = parse_args()

    # set the device to use for training
    device = Utils.get_device(args.device)

    # limit the cores used by torch
    torch.set_num_threads(args.max_n_cores)
    torch.set_num_interop_threads(args.max_n_cores)

    # create a random generator with the input random seed for reproducibility
    random_generator = np.random.default_rng(args.random_seed)

    # --------------------------------------------------- SIMULATION I ---------------------------------------------------

    # run simulation I, if not excluded
    if not args.no_sim_I:

        # path where to save the results of simulation I
        output_dir_sim_I = os.path.join(args.output_dir, 'I')

        # consider binary, branching and linear trees
        simulation_I(output_dir_sim_I, generate_data_simulation_I, 'I_bin_bran_lin', n_nodes=args.n_nodes_I, n_samples=args.n_samples, n_random_ops_list=args.n_random_ops, n_experiments=args.n_exp, node_encoding_type=args.node_encoding_type, random_generator=random_generator, device=device, include_baseline=False)
        
        # consider only branching and linear trees
        simulation_I(output_dir_sim_I, partial(generate_data_simulation_I, no_binary=True), 'I_bran_lin', n_nodes=args.n_nodes_I, n_samples=args.n_samples, n_random_ops_list=args.n_random_ops, n_experiments=args.n_exp, node_encoding_type=args.node_encoding_type, random_generator=random_generator, device=device, include_baseline=False)

    # --------------------------------------------------- SIMULATION II ---------------------------------------------------

    # run simulation II, if not excluded
    if not args.no_sim_II:

        # path where to save the results of simulation II
        output_dir_sim_II = os.path.join(args.output_dir, 'II')

        # consider binary, branching and linear trees
        simulation_II(output_dir_sim_II, partial(generate_data_simulation_II, n_samples=args.n_samples, random_generator=random_generator), 'II_bin_bran_lin', n_nodes_list=args.n_nodes_list, n_exp=args.n_exp, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)
        
        # consider only branching and linear trees
        simulation_II(output_dir_sim_II, partial(generate_data_simulation_II, n_samples=args.n_samples, random_generator=random_generator, no_binary=True), 'II_bran_lin', n_nodes_list=args.n_nodes_list, n_exp=args.n_exp, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)

    # --------------------------------------------------- SIMULATION III ---------------------------------------------------

    # run simulation III, if not excluded
    if not args.no_sim_III:
        other_simulations(args.output_dir, generate_data_simulation_III, 'III', n_nodes_list=args.n_nodes_list, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)

    # --------------------------------------------------- SIMULATION IV ---------------------------------------------------

    # run simulation IV, if not excluded
    if not args.no_sim_IV:
        other_simulations(args.output_dir, generate_data_simulation_IV, 'IV', n_nodes_list=args.n_nodes_list, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)
        
    # --------------------------------------------------- SIMULATION V ---------------------------------------------------

    # run simulation V, if not excluded
    if not args.no_sim_V:

        # path where to save the results of simulation V
        output_dir_sim_V = os.path.join(args.output_dir, 'V')

        # consider exclusive, ancestry and the inverse of ancestry relations
        other_simulations(output_dir_sim_V, generate_data_simulation_V, 'V_exclusive_vs_ad_vs_da', n_nodes_list=args.n_nodes_list, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)
        
        # consider only exclusive and ancestry relations
        other_simulations(output_dir_sim_V, partial(generate_data_simulation_V, no_descendants=True), 'V_exclusive_vs_ad', n_nodes_list=args.n_nodes_list, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)
        
        # consider only ancestry and its inverse relations
        other_simulations(output_dir_sim_V, partial(generate_data_simulation_V, no_exclusives=True), 'V_ad_vs_da', n_nodes_list=args.n_nodes_list, node_encoding_type=args.node_encoding_type, random_seed=args.random_seed, device=device, include_baseline=False)