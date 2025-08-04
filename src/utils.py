import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def remove_long_trees(dataset, max_length=9):
    """
    Removes trees in the input dataset with more than max_length edges.

    Parameters:
    - dataset: numpy array with patients. Each patient contains a list of plausible trees.
    - max_length: maximum number of edges that a tree can have.

    Returns:
    - new_dataset: numpy array which is a version of the input dataset without trees longer than max_length.
    """

    # dataset without long trees
    new_dataset = []

    # add to the new dataset only trees with less edges than max_length
    for patient in dataset:
        new_patient = []
        for tree in patient:
            if len(tree) <= max_length:
                new_patient.append(tree)
        
        # append the patient only if it contains at least a valid tree
        if len(new_patient) > 0:
            new_dataset.append(new_patient)
    
    # convert the list of lists into a numpy array, as the original dataset
    new_dataset = np.array(new_dataset, dtype='object')

    return new_dataset

def train_test_split(dataset, p=0.8, rd_seed=None):
    """
    Splits the input dataset into two disjoint lists of patients: a training set and a test set.

    Parameters:
    - dataset: dataset to be split.
    - p: ratio of patients in the input dataset to be included in the training set.
    
    Returns:
    - training_set: training set.
    - test_set: test set.
    """

    # set the random seed, if required
    if rd_seed is not None:
        np.random.seed(rd_seed)

    # number of patients to be included in the training set
    n_train = int(p * len(dataset))

    # consider a shuffled version of data to then easily get a random partition
    shuffled_dataset = np.copy(dataset)
    np.random.shuffle(shuffled_dataset)

    # training and test sets
    training_set = shuffled_dataset[:n_train]
    test_set = shuffled_dataset[n_train:]

    return training_set, test_set   

def get_dictionary_tree(tree):
    """
    Represents the input tree as a dictionary tree_dic.

    Parameters:
    - tree: array representing the input tree as a list of edges. Each edge is a list with the labels of the two nodes it links.

    Returns:
    - tree_dic: dictionary with two keys: 'nodes' and 'edges'.
                                        tree_dic['nodes']: dictionary with an integer univoquely identifying a node as key for each node and the label of the node as value.
                                        tree_dic['edges']: list of edges in the tree. Each edge is a list with the integer ids of the two nodes it links.

    """

    # map each node in the tree to an integer id, with the root node that has id 0
    mapping = node_id_mapping(tree)

    # initialize the dictionary that will represent the input tree
    tree_dic = {
        'nodes': {},
        'edges': []
    }

    # fill tree_dic with the edge list, where are nodes represented as integer ids
    for edge in tree:
        tree_dic['edges'].append([mapping[edge[0]], mapping[edge[1]]])

    # replace the label of the root node with "root", so to have consistency among different trees
    root_label = find_root(tree)
    del mapping[root_label]
    mapping['root'] = 0

    # fill tree_dic with the dictionary that has an entry for each node, where the keys are the integer ids and the values are the labels
    for node_label, node_id in mapping.items():
        tree_dic['nodes'][node_id] = node_label

    return tree_dic

def convert_into_dictionaries(patients):
    """
    Converts the trees in the input numpy array into dictionaries, as required by our model.

    Parameters:
    - patients: numpy array containing the trees to be converted.
                The numpy array contains patients.
                Each patient is a list of trees.
                A tree is represented as a list of edges.
                Each edge is a list with the labels of the two nodes it links.
    
    Returns:
    - patients_dic: version of patients with each tree represented as a dictionary.
                    trees_dic is a list of patients.
                    Each patient is a list of trees.
                    Each tree is a dictionary with two keys: 'nodes' and 'edges'.
                    'nodes' is a dictionary with an integer univoquely identifying a node as key for each node and the label of the node as value.
                    'edges' is a list of edges in the tree. Each edge is a list with the integer ids of the two nodes it links.
    """
    
    # initialize the list of patients
    patients_dic = []

    # fill the list with patients, where each patient has its trees represented as dictionaries
    for patient in patients:
        curr_patient_trees = []
        for tree in patient:
            curr_patient_trees.append(get_dictionary_tree(tree))
        patients_dic.append(curr_patient_trees)
    
    return patients_dic

def save_dataset_txt(patients, save_path):
    """
    Saves the input dataset into a .txt file with the format required by our model.

    Parameters:
    - patients: list containing graphs organized into patients.
                Each patient is a list of graphs.
                A graph is represented as a dictionary with two keys: 'nodes' and 'edges'.
    - save_path: path to the file where to save the input dataset.
    """

    # open the file in writing mode
    with open(save_path, 'w') as file:

        # number of patients in the input dataset
        n_patients = len(patients)

        # write the number of patients in the first line
        file.write(f'{n_patients} patients\n')

        # iterate through patients
        for i in range(n_patients):

            # number of graphs for the current patient
            n_graphs = len(patients[i])

            # write the number of graphs
            file.write(f'{n_graphs} graphs for patient {i}\n')

            # iterate through graphs
            for j in range(n_graphs):

                # write the number of nodes
                file.write(f'{len(patients[i][j]["nodes"].keys())} nodes in graph {j}\n')

                # write each node in a different line with in the format "node_id node_label" such that lines are ordered by key
                sorted_node_ids = sorted(patients[i][j]['nodes'].keys())
                for node_id in sorted_node_ids:
                    file.write(f'{node_id} {patients[i][j]["nodes"][node_id]}\n')

                # write the number of edges
                file.write(f'{len(patients[i][j]["edges"])} edges in graph {j}\n')

                # write each edge in a different line
                for edge in patients[i][j]['edges']:
                    file.write(f'{edge[0]} {edge[1]}\n')

def save_dataset_txt_survival(patients_dic, save_path):
    """
    Saves the input dataset into a .txt file with the format required by our model.

    Parameters:
    - patients_dic: dictionary, where each entry is a patient, with the id as key and the list of TumorGraphs as value.
    - save_path: path to the file where to save the input dataset.
    """

    # open the file in writing mode
    with open(save_path, 'w') as file:

        # write the number of patients in the dataset
        file.write(f'{len(patients_dic)} patients\n')

        # iterate through patients to save them
        for patient_id, patient in patients_dic.items():

            # write the id and number of graphs for the current patient
            file.write(f'{len(patient)} graphs for patient {patient_id}\n')

            # iterate through the graphs for the current patient
            for graph in patient:

                # write the number of nodes in the graph
                file.write(f'{graph.n_nodes()} nodes\n')

                # iterate through the nodes in the graph and save them
                for node in graph.nodes:

                    # itertate through the labels of the node and save them
                    labels = node.labels
                    file.write(f'{node.id} {labels[0]}')
                    labels = labels[1:]
                    for label in labels:
                        file.write(f',{label}')
                    file.write('\n')
                
                # write the number of edges in the graph
                file.write(f'{graph.n_edges()} edges\n')

                # iterate through the edges in the graph and save them
                for edge in graph.edges:
                    file.write(f'{edge[0]} {edge[1]}\n')

def collapse_gene_level(dic_graphs):
    """
    Collapses mutations at gene level for all graphs in the input dataset.

    Parameters:
        - patients_dic: list of patients where each patient is a list of graphs.
                        A graph is represented as a dictionary with two keys: 'nodes' and 'edges'.
                        'nodes' is a dictionary with an integer univoquely identifying a node as key for each node and the label of the node as value.
                        'edges' is a list of edges in the graph. Each edge is a list with the integer ids of the two nodes it links.
    
    Returns:
        - gene_patients_dic: version of patients_dic with mutations collapsed at gene level.
                                The mutations are collapsed at gene level by truncating at the first occurrence of an underscore ('_') or a period ('.').
    """

    # new version of the dataset that will be returned
    gene_patients_dic = []

    # iterate through patients
    for patient in dic_graphs:
        
        # new version of the current patient
        new_patient = []

        # iterate through graphs of the current patient
        for graph in patient:

            # new version of the current graph of the current patient
            new_graph = {
                'nodes': {},
                'edges': []
            }

            # fill new_graph with the nodes, where the labels are collapsed at gene level
            for node_id, node_label in graph['nodes'].items():
                new_node = ""
                for i in range(len(node_label)):
                    if node_label[i] == "_" or node_label[i] == ".":
                        break
                    else:
                        new_node += node_label[i]
                new_graph['nodes'][node_id] = new_node
            
            # since the node ids do not change, the edges can be directly copied
            new_graph['edges'] = graph['edges']
            
            # add new_graph to new_patient
            new_patient.append(new_graph)
        
        # add new_patient to gene_patients_dic
        gene_patients_dic.append(new_patient)
    
    return gene_patients_dic

def get_all_nodes(tree):
    """
    Returns all the nodes in the input tree.

    Parameters:
    - tree: list of edges representing the tree.
            Each edge is a list with the labels of the two nodes it links.
    
    Returns:
    - nodes: list of all the nodes in the input tree.
    """

    # list that will contain all nodes in the tree
    nodes = []

    # iterate trìhrough the edges in the tree to fill nodes
    for edge in tree:
        for node in edge:
            if node not in nodes:
                nodes.append(node)

    return nodes        

def find_root(tree):
    """
    Finds the root of the input tree.
    The root of a tree is the only node in the tree with no parent.

    Parameters:
    - tree: list of edges representing the tree.
            Each edge is a list with the labels of the two nodes it links.

    Returns:
    - nodes[0]: the only node in the tree with no parent, i.e., the root of the tree.
    """

    # get all nodes in the tree
    nodes = get_all_nodes(tree)

    # iterate through the edges in the tree and remove from the list of nodes those that appear as children
    for edge in tree:
        if edge[1] in nodes:
            nodes.remove(edge[1])
    
    # the root is the only node left in the list of nodes
    return nodes[0]

def node_id_mapping(tree):
    """
    Creates a mapping that assigns an integer id to each node in the input tree.
    The root node is assigned id 0.

    Parameters:
    - tree: array representing the input tree as a list of edges. Each edge is a list with the labels of the two nodes it links.

    Returns:
    - mapping: dictionary with an entry for each node.
            Each node has its label as key and the integer id as value.
            The root node has id 0.
    """

    # get all nodes in the tree
    nodes = get_all_nodes(tree)

    # find the root node in the tree
    root = find_root(tree)

    # create a mapping that assigns each node in the tree an integer id, with the root having id 0
    mapping = {root: 0}
    curr_id = 0
    for node in nodes:
        if node != root:
            curr_id += 1
            mapping[node] = curr_id

    return mapping

def one_row_per_patient(clinical_df, time_label, event_label):
    """
    Condenses data related to multiple samples from the same patient into a single row.
    Specifically, if there are multiple rows with the same value of Patient_ID, then just the one with value of time_label that is the largest is kept.

    Parameters:
    - clinical_df: pandas DataFrame containing the clinical data.
                clinical_df['Patient_ID'] contains patient ids and clinical_df[labels] contains the clinical labels.
    - time_label: string with the name of the column containing the time of the event.
    - event_label: string with the name of the column containing whether the event happened or not.

    Returns:
    - consistent_df: pandas DataFrame containing the clinical data of patients with just one row per patient.
    """

    # create a copy of the input dataframe
    consistent_df = clinical_df.copy()

    # iterate through the unique patient ids in the dataset
    for patient_id in clinical_df['Patient_ID'].unique():
        
        # get the rows of the current patient
        patient_rows = clinical_df[clinical_df['Patient_ID'] == patient_id]

        # get the row with the largest time of the event
        max_time = patient_rows[time_label].max()
        max_time_row = patient_rows[patient_rows[time_label] == max_time]

        # keep just the row with the largest time of the event
        consistent_df = consistent_df[consistent_df['Patient_ID'] != patient_id]
        consistent_df = pd.concat([consistent_df, max_time_row])
    
    return consistent_df

def remove_unknown_values(clinical_df, labels):
    """
    Removes patients from the input dataset if their values are unknown or if they contain NA values.

    Parameters:
    - clinical_df: pandas DataFrame containing the clinical data.
                clinical_df['Patient_ID'] contains patient ids and clinical_df[labels] contains the clinical labels.
    - labels: list of strings with the name of the columns containing the clinical labels.

    Returns:
    - consistent_df: pandas DataFrame containing the clinical data of patients with known clinical labels.
    """

    # create a copy of the input dataframe
    consistent_df = clinical_df.copy()

    # remove patients with NA values
    consistent_df = consistent_df.dropna()

    # remove patients with unknown value of the input labels
    for lab in labels:
        consistent_df = consistent_df[consistent_df[lab] != 'Unknown']
        consistent_df = consistent_df[consistent_df[lab] != 'unk']

    return consistent_df

def intersect_ids(phylogenies, clinical_data):
    """
    Returns the input phylogenies and clinical data with only the patients whose ids are present in both datasets.

    Parameters:
    - phylogenies: dictionary with patient ids as keys and lists of TumorGraphs as values.
    - clinical_data: pandas DataFrame with patient ids in the column 'Patient_ID'.

    Returns:
    - phylogenies: dictionary with patient ids as keys and lists of TumorGraphs as values, containing only the patients whose ids are present in both datasets.
    - clinical_data: pandas DataFrame containing the clinical data of patients whose ids are present in both datasets.
    """

    # get the patient ids present in both datasets
    ids = set(phylogenies.keys()).intersection(set(clinical_data['Patient_ID'].unique()))

    # sort the ids so to enhance reproducibility
    ids = sorted(list(ids))

    # keep only the patients whose ids are present in both datasets and return the updated datasets
    return {patient_id: phylogenies[patient_id] for patient_id in ids}, clinical_data[clinical_data['Patient_ID'].isin(ids)]

def remove_patients_with_uncertain_phylogeny(patients_dic, max_n_graphs):
    """
    Removes from the input dataset the patients with more than max_n_graphs phylogenetic graphs.

    Parameters:
    - patients_dic: dictionary with patient ids as keys and lists of TumorGraphs as values.
    - max_n_graphs: integer, maximum number of phylogenetic graphs a patient can have.

    Returns:
    - processed_patients_dic: dictionary with patient ids as keys and lists of TumorGraphs as values, containing only the patients with at most max_n_graphs phylogenetic graphs.
    """
        
    # remove patients with more than max_n_graphs phylogenetic graphs and return them
    return {patient_id: graphs for patient_id, graphs in patients_dic.items() if len(graphs) <= max_n_graphs}

def analyze_label_frequency(clinical_df, clinical_label, info_label):
    """
    Analyzes the distribution of values taken by the input value in the input dataframe.

    Parameters:
    - clinical_df: pandas DataFrame containing the clinical data.
                clinical_df['Patient_ID'] contains patient ids and clinical_df[label] contains the clinical label value.
    - clinical_label: string with the name of the column containing the clinical label.
    - info_label: string with the name of the dataset to be analyzed.

    Returns:
    - counts: pandas DataFrame with the count for each value the input label takes in the input dataframe.
    """

    # sort the rows of the dataframe by lexicographic order of the class
    sorted_clinical_df = clinical_df.sort_values(by=clinical_label)

    # count the number of occurrences of each value of the input label in the dataset
    counts = sorted_clinical_df[clinical_label].value_counts()

    # compute also the frequency of each value
    frequencies = sorted_clinical_df[clinical_label].value_counts(normalize=True)

    # plot the distribution of the input clinical_label
    sns.set_theme('paper')
    sns.countplot(data=sorted_clinical_df, x=clinical_label)

    plt.title(f'Distribution of {clinical_label} in {info_label}')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.show()

    # print the count and frequency of each value of the input label
    print(f'\n{"-"*20} Class frequencies of {clinical_label} label in {info_label} {"-"*20}\n')
    print('Class | Count | Frequency')
    for value, count in counts.items():
        print(f'{value} | {count} | {frequencies[value]: .2f}')
    
    return counts

def get_device(device=None):
    """
    Get the device to use for tensor operations.

    Parameters:
    - device: string with the name of the device to use. Can be 'cuda', 'mps' or 'cpu'.
                If None, the device is selected among 'cuda', 'mps' and 'cpu' in decreasing order of preference.

    Returns:
    - torch.device to be used for tensor operations.
        Device is selected among 'cuda', 'mps' and 'cpu' in decreasing order of preference if None is provided as input.
    """
    if device in ['cuda', 'mps', 'cpu']:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def flatten_list_of_lists(list_of_lists):
    """
    Flatten a list of lists into a single list.

    Parameters:
    - list_of_lists: list of lists to be flattened.

    Returns:
    - flat_list: single list with all elements in the input list of lists.
                    Elements are ordered in the following way:
                    [list_of_lists[0][0], list_of_lists[0][1], ..., list_of_lists[0][len(list_of_lists[0]) - 1], list_of_lists[1][0], ...,
                    list_of_lists[1][len(list_of_lists[1]) - 1], ..., list_of_lists[len(list_of_lists) - 1][len(list_of_lists[len(list_of_lists) - 1]) - 1].
    """

    return [elem for sublist in list_of_lists for elem in sublist]

def get_n_items_list_of_lists(list_of_lists):
    """
    Get the overall number of items in the input list of lists.

    Parameters:
    - list_of_lists: list of lists.

    Returns:
    - n_items_list: int representing the overall number of items in list_of_lists.
    """

    return sum([len(sublist) for sublist in list_of_lists])

def select_loss(string_loss):
    """
    Returns the loss function identified by the input string.

    Parameters:
    - string_loss: string with the name of the loss function to use.

    Returns:
    - loss function identified by the input string.
    """

    if string_loss == 'MSE_loss':
        return torch.nn.MSELoss()
    elif string_loss == 'MAE_loss':
        return torch.nn.L1Loss()
    
    raise ValueError('Invalid loss function name.')

def select_optimizer(string_opt):
    """
    Returns the optimizer identified by the input string.

    Parameters:
    - string_opt: string with the name of the optimizer to use.

    Returns:
    - optimizer identified by the input string.
    """

    if string_opt == 'Adam':
        return torch.optim.Adam
    elif string_opt == 'SGD':
        return torch.optim.SGD
    
    raise ValueError('Invalid optimizer name.')

def merge_clustering_clinical_data(cluster_labels, clinical_data):
    """
    Merges cluster labels for different clustering sizes with clinical data.

    Parameters:
    - cluster_labels: dictionary with values of k as keys and arrays with cluster labels for embeddings as values.
                        Each array is sorted by patient id.
    - clinical_data: pandas dataframe with clinical data for patients. It must contain column 'Patient_ID'.

    Returns:
    - complete_df: pandas dataframe with the clustering and clinical data merged.
                    It has columns: 'Patient_ID', 'K_2', 'K_3', ..., 'K_n', where K_i is the cluster label assigned
                    to the patient for the clustering with size i. It also contains all the columns in clinical_data.
    """

    # copy the input dataframe with clinical data
    complete_df = clinical_data.copy()

    # sort the dataframe by patient id
    complete_df.sort_values(by='Patient_ID', inplace=True)

    # get an array with the sorted patient ids
    sorted_ids = complete_df['Patient_ID'].values

    # create a dictionary with keys 'Patient_ID', 'K_2', 'K_3', ..., 'K_n', where K_i is the cluster label assigned to the patient for the clustering with size i
    clusterings_data = []
    for i, id in enumerate(sorted_ids):
        curr_row = {'Patient_ID': id}
        for k in cluster_labels.keys():
            curr_row[f'K_{k}'] = cluster_labels[k][i]
        clusterings_data.append(curr_row)
    
    # convert the dictionary with clustering data into a pd dataframe
    clusterings_df = pd.DataFrame(clusterings_data)

    # merge the two dataframes by performing a inner join on column 'Patient_ID'
    complete_df = complete_df.merge(clusterings_df, how='inner', on='Patient_ID')

    return complete_df

def listdir_nohidden(path):
    """
    Lists all subdirectories in the input directory, excluding hidden subdirectories.

    Parameters:
    - path: string with the path to the directory to be listed.

    Returns:
    - generator yielding the names of all subdirectories in the input directory, excluding hidden subdirectories.
    """

    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f