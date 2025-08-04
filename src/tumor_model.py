import os
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import numpy as np
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import utils as Utils
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm
import networkx as nx
import random as rd
import torch.utils
import torch.utils.data
from torch_geometric.data import Data
import copy

class TumorNode:
    """
    Class to store and use a node of a TumorGraph object.
    """

    def __init__(self, node_id, node_labels=["empty"]):
        """
        Constructor.

        Parameters:
        - node_id: integer id that univoquely identifies the node in the graph it belongs to.
                   It must be an integer in [0, n_nodes - 1], where n_nodes is the number of nodes in the graph it belongs to.
                   If the current node is the root, i.e., if it represents the germline subpopulation of cells, then its id must be 0.
        - node_labels: list of strings with the labels to be assigned to the node.
                       If the node represents the germline subpopulation of cells, then the list of labels must be ["root"].
                       If the node does not have any label, then its list of labels must be ["empty"].
        """

        # node id
        self.id = node_id

        # node labels
        self.labels = node_labels

class TumorGraph:
    """
    Class to store and use graphs representing cancer clonal evolution.
    """

    def __init__(self, nodes=None, edges=None):
        """
        Constructor.

        Parameters:
        - nodes: list of TumorNode objects. Can be None, in which case an empty TumorGraph will be created.
        - edges: list of edges. An edge is a tuple with the ids of the two TumorNode objects it links.
                 Since edges are directed, order matters.
                 Can be None, in which case, the TumorGraph will not have any edge.
        """

        # nodes
        self.nodes = []
        if nodes is not None:
            self.add_node_list(nodes)

        # edges
        self.edges = []
        if edges is not None:
            self.add_edge_list(edges)

    def n_nodes(self):
        """
        Computes the number of TumorNodes in the TumorGraph.

        Returns:
        - n_nodes: number of TumorNodes in the TumorGraph.
        """

        return len(self.nodes)

    def n_edges(self):
        """
        Computes the number of edges in the TumorGraph.

        Returns:
        - n_edges: number of edges in the TumorGraph.
        """

        return len(self.edges)

    def remove_node(self, id):
        """
        Removes the TumorNode with the input id from the TumorGraph and all edges in which it appears.

        Parameters:
        - id: id of the TumorNode to be removed from the graph.

        Returns:
        - 0 if the TumorNode was successfully removed, -1 if no TumorNode with the input id is in the TumorGraph.
        """

        # if there is no TumorNode with the input id, then return -1
        if id not in self.get_node_ids():
            return -1

        # initialize the new lists of TumorNodes and edges
        new_nodes = []
        new_edges = []

        # add all TumorNodes with id different from the input one to new_nodes
        for node in self.nodes:
            if node.id != id:
                new_nodes.append(node)
        
        # add all edges in which the input id does not appear to new_edges
        for edge in self.edges:
            if edge[0] != id and edge[1] != id:
                new_edges.append(edge)
        
        # update the list of TumorNodes and edges of the TumorGraph
        self.nodes = new_nodes
        self.edges = new_edges

        return 0

    def remove_edge(self, edge):
        """
        Removes the input edge from the TumorGraph.
        If the input edge is not in the TumorGraph, then nothing happens.

        Parameters:
        - edge: tuple with the ids of the TumorNodes it links. Edge to be removed from the TumorGraph.

        Returns:
        - 0 if the edge was successfully removed, -1 if the edge not present in the TumorGraph.
        """

        # if the edge not present in the TumorGraph, then return -1
        if edge not in self.edges:
            return -1

        # otherwise, remove it from the TumorGraph
        self.edges.remove(edge)

        return 0

    def add_node(self, node):
        """
        Adds the input TumorNode to the TumorGraph.
        If a TumorNode with the same id already exists in the TumorGraph, then it is replaced by the input one.

        Parameters:
        - node: TumorNode to be added to the TumorGraph.
        """

        # if a TumorNode with the same id of the one to be added is already in the TumorGraph, then replace it by the input one
        for i, curr_node in enumerate(self.nodes):
            if curr_node.id == node.id:
                self.nodes[i] = node
                return
        
        # otherwise, add the TumorNode to the TumorGraph
        self.nodes.append(node)

    def add_edge(self, edge):
        """
        Adds the input edge to the TumorGraph.
        If the edge contains ids of TumorNodes not in the TumorGraph, then an empty TumorNode is created and added to the TumorGraph for each id not present.
        If the edge is already present in the TumorGraph, then nothing happens.

        Parameters:
        - edge: tuple with the ids of the two TumorNodes it links. Order matters since edges are directed.
        """

        # if the edge is already present in the TumorGraph, then simply return
        if edge in self.edges:
            return

        # create a TumorNode for each id in the edge not present in the TumorGraph
        ids = self.get_node_ids()
        for node_id in edge:
            if node_id not in ids:
                self.nodes.append(TumorNode(node_id))
        
        # add the edge to the TumorGraph
        self.edges.append(edge)

    def add_node_list(self, nodes):
        """
        Adds the input list of TumorNodes to the TumorGraph object.
        TumorNodes already present in the TumorGraph will be replaced.

        Parameters:
        - nodes: list of TumorNode objects to be added.
        """

        # add to the TumorGraph each TumorNode in nodes
        for node in nodes:
            self.add_node(node)
    
    def add_edge_list(self, edges):
        """
        Adds the input list of edges to the TumorGraph.
        Edges already present will not be added.

        Parameters:
        - edges: list of edges to be added to the TumorGraph. An edge is a tuple with the ids of the two TumorNodes it connects.
        """

        # add each edge in edges to the TumorGraph
        for edge in edges:
            self.add_edge(edge)
    
    def get_node_ids(self, sort=False):
        """
        Returns all the ids of the TumorNode objects in the TumorGraph.

        Parameters:
        sort: boolean indicating whether the ids must be returned in ascending order or not.

        Returns:
        - ids: list with the ids of the nodes in the TumorGraph.
        """

        # initialize the list of ids
        ids = []

        # fill the list with all the ids of the nodes in the TumorGraph
        for node in self.nodes:
            ids.append(node.id)
        
        # return the ids sorted, if required
        if sort:
            return sorted(ids)
        return ids
    
    def get_unique_labels(self):
        """
        Returns the set of labels appearing in the TumorNodes of the TumorGraph.
        Each label is reported once.

        Returns:
        labels: set with all node labels appearing in the TumorGraph.
        """

        # initialize the set of labels appearing in the TumorNodes of the TumorGraph
        labels = set()

        # iterate through the TumorNodes in the graph so to add new labels in the set
        for node in self.nodes:
            labels.update(node.labels)
        
        return labels
    
    def to_DiGraph(self):
        """
        Returns the TumorGraph as a networkx DiGraph object.

        Returns:
        - graph_nx: networkx DiGraph object representing the TumorGraph.
        """

        # create a directed graph
        graph_nx = nx.DiGraph()

        # add nodes to the graph
        for node in self.nodes:
            graph_nx.add_node(node.id, labels=node.labels)
        
        # add edges to the graph
        graph_nx.add_edges_from(self.edges)

        return graph_nx

class TumorDataset:
    """
    Class to manage a dataset of TumorGraph objects organized into patients.
    """

    def __init__(self, data):
        """
        Constructor.

        Parameters:
        - data: input data. Can be either the path to a .txt file with tumor data or a list of patients, where a patient is a list of TumorGraphs.
        """
        
        # initialize the dataset with the input data
        if isinstance(data, (str, os.PathLike)):
            self.dataset = self.read_txt(data)
        else:
            self.dataset = data

    def read_txt(self, dataset_path):
        """
        Loads data from a .txt file with proper format.

        Parameters:
        - dataset_path: path to the .txt file containing data to load.

        Returns:
        - data: list of patients. Each patient is a list of TumorGraph objects
        """

        # initialize the list that will contain the data to load
        data = []

        # open the file in read mode
        with open(dataset_path, "r") as file:

            # the first line contains the number of patients
            n_patients = int(file.readline().split()[0])

            # iterate through patients
            for i in range(n_patients):

                # append a list related to a new patient
                data.append([])

                # the first line related to a patient has the number of graphs it contains
                n_graphs = int(file.readline().split()[0])

                # iterate through the graphs for the current patient
                for j in range(n_graphs):
                    
                    # initialize the current TumorGraph for the current patient
                    data[i].append(TumorGraph())

                    # the first line related to a graph has the number of nodes in the graph
                    n_nodes = int(file.readline().split()[0])

                    # add all nodes, represented as TumorNodes to the current TumorGraph, where node labels are comma separated
                    for k in range(n_nodes):
                        node = file.readline().split()
                        node_id = int(node[0])
                        node_labels = node[1].split(",")
                        data[i][j].add_node(TumorNode(node_id=node_id, node_labels=node_labels))
                    
                    # the next line contains the number of edges in the current graph
                    n_edges = int(file.readline().split()[0])

                    # add all edges, represented as tuple with the ids of the TumorNodes it links
                    for l in range(n_edges):
                        edge = file.readline().split()
                        data[i][j].add_edge((int(edge[0]), int(edge[1])))
                

        
        return data
    
    def n_patients(self):
        """
        Computes the number of patients in the dataset.

        Returns:
        - len(self.dataset): number of patients in the dataset.
        """

        return len(self.dataset)
    
    def n_graphs(self):
        """
        Computes the overall number of graphs in the dataset.

        Returns:
        - n_graphs: overall number of graphs in the dataset.
        """

        # iterate through patients to compute the overall number of trees in the dataset
        n_graphs = 0
        for patient in self.dataset:
            n_graphs += len(patient)
        
        return n_graphs

    def n_graphs_patient(self, i):
        """
        Computes the number of graphs for the patient in position i in the dataset.

        Parameters:
        - i: index of the patient in the dataset.

        Returns:
        - len(self.dataset[i]): number of graphs for patient i in the dataset.
        """

        return len(self.dataset[i])

    def node_labels(self):
        """
        Returns all node labels appearing in self.dataset.
        Node labels "root", "unknown" and "empty" are added to the set even in case they do not appear in the self.dataset.

        Returns:
        - labels: set with all labels appearing in self.dataset plus "root", "unknown" and "empty".
        """

        # initialize the set of labels with "root", "unknown" and "empty"
        labels = {"root", "unknown", "empty"}

        # iterate through all nodes in the dataset and add not already found labels
        for patient in self.dataset:
            for graph in patient:
                labels.update(graph.get_unique_labels())
        
        return labels

    def labels_counts(self):
        """
        Computes the number of patients in which each node label present in self.dataset appears.

        Returns:
        - labels_counts_dic: dictionary with all node labels in self.dataset as keys and the corresponding number of patients in which they occur as values.
        """

        # compute the set of all node labels appearing in the dataset
        labels = self.node_labels()

        # initialize the dictionary
        labels_counts_dic = {}

        # iterate through all labels in the dataset
        for label in labels:

            # initialiaze the number of patients that have the current label
            labels_counts_dic[label] = 0

            # compute the number of patients with the current label
            for patient in self.dataset:
                for graph in patient:
                    if label in graph.get_unique_labels():
                        labels_counts_dic[label] += 1
                        break
    
        return labels_counts_dic

    def node_labels_freq(self, n=2):
        """
        Returns all node labels appearing at least in n patients in self.dataset.
        A lable is considered to appear in a patient if it is present in a node of at least one of the graphs for the considered patient.
        
        Parameters:
        - n: minimum number of patients with at least one node in one graph with a given label for it to be included in the set of labels.

        Returns:
        - labels: set with all labels appearing in at least n patients in self.dataset.
        """

        # compute the number of patients in which each label appears
        labels_counts_dic = self.labels_counts()

        # return only the labels appearing in at least n patients
        return set(key for key in labels_counts_dic.keys() if labels_counts_dic[key] >= n)
    
    def remove_infreq_labels(self, threshold):
        """
        Removes all node labels appearing in less than threshold patients in self.dataset.
        When a label is removed from a node, the node is assigned label "empty" if it does not contain any other label, representing a node with no label.
        Also the node label "unknown", regrdless of the number of times it appears in the dataset, is replaced with "empty".

        Parameters:
        - threshold: minimum number of patients in which a node label must appear not to be removed from self.dataset.
        """

        # compute the number of patients in which each label appears
        labels_counts_dic = self.labels_counts()

        # set the count of "unknown" to 0, so not to keep it
        labels_counts_dic["unknown"] = 0

        # remove infrequent labels from the dataset
        for i in range(self.n_patients()):
            for j in range(self.n_graphs_patient(i)):
                for k, node in enumerate(self.dataset[i][j].nodes):
                    for label in node.labels:
                        if labels_counts_dic[label] < threshold:
                            self.dataset[i][j].nodes[k].labels = [l for l in self.dataset[i][j].nodes[k].labels if l != label]
                    if len(self.dataset[i][j].nodes[k].labels) == 0:
                        self.dataset[i][j].nodes[k].labels = ["empty"]
        
    def remove_large_graphs(self, max_n_edges=10):
        """
        Removes the graphs with more than max_n_edges edges from self.dataset.
        If a patient becomes empty due to the removal of all its graphs, then it is removed from self.dataset.

        Parameters:
        - max_n_edges: maximum number of edges that a graph can have so to be kept in self.dataset
        """

        # version of the self.dataset that will not have large graphs
        new_dataset = []

        # iterate through all graphs and keep only those smaller than max_n_edges
        for patient in self.dataset:
            new_patient = []
            for graph in patient:
                if graph.n_edges() <= max_n_edges:
                    new_patient.append(graph)
            
            # append new_patient only if it is not empty
            if len(new_patient) > 0:
                new_dataset.append(new_patient)

        # update self.dataset with new_dataset
        self.dataset = new_dataset

    def remove_uncertain_patients(self, max_n_graphs=20):
        """
        Removes uncertain patients from self.dataset.
        A patient is considered uncertain if it has a number of graphs larger than max_n_graphs.
        
        Parameters:
        - max_n_graphs: maximum number of graphs that a patient can have so to be kept in self.dataset.
        """

        # new version of the dataset without uncertain patients
        new_dataset = []

        # iterate through patients and keep only those with less than max_n_graphs graphs
        for i in range(self.n_patients()):
            if self.n_graphs_patient(i) < max_n_graphs:
                new_dataset.append(self.dataset[i])
        
        # update self.dataset with the version with no uncertain patient
        self.dataset = new_dataset

    def sample_one_graph_per_patient(self, rd_seed=None):
        """
        Samples uniformly at random one graph per patient from self.dataset.
        The function updates self.dataset such dataset each patient will be left with just one graph chosen uniformly at random among those previously contained by the patient.
        Notice that each patient will still be a list of graphs, but with only one element, that is, the sampled graph.

        Parameters:
        - rd_seed: random seed for sampling reproducibility.
        """

        # set the random seed, if required
        if rd_seed is not None:
            rd.seed(rd_seed)

        # new version of the dataset with only one graph per patient
        new_dataset = []

        # iterate through patients and sample just one graph uniformly at random
        for patient in self.dataset:
            new_dataset.append([patient[rd.randint(0, len(patient) - 1)]])
        
        # update self.dataset with the new version of the dataset
        self.dataset = new_dataset

    def replace_label_set(self, known_labels, replace_with="unknown"):
        """
        Replaces all node labels in all graphs for all patients in self.dataset that are not in known_labels with the label replace_with.
        Labels "root", "unknown" and "empty" are added to known_labels and not replaced when found.

        Parameters:
        - known_labels: set with known node labels. Node labels in this set will not be replaced with label replace_with.
                        Also "root", "empty" and "unknown" will not br replaced.
        """

        # add to the set of labels to keep also "root", "empty", "unknown" and replace_with, if not already present
        labels_to_keep = known_labels.copy()
        labels_to_keep.update({"root", "empty", "unknown", replace_with})

        # iterate through all nodes in self.dataset and replace the node labels not in knwon_labels with label replace_with
        for i in range(self.n_patients()):
            for j in range(self.n_graphs_patient(i)):
                for k, node in enumerate(self.dataset[i][j].nodes):
                    for label in node.labels:
                        if label not in labels_to_keep:
                           self.dataset[i][j].nodes[k].labels = [l for l in self.dataset[i][j].nodes[k].labels if l != label]
                           self.dataset[i][j].nodes[k].labels.append(replace_with)
    
    def to_dataset_DiGraphs(self):
        """
        Returns a list of patients, where each patient is a list of networkx DiGraph objects.
        Each TumorGraph in the TumorDataset is converted into a networkx DiGraph.

        Returns:
        - dataset_nx: list of patients, where each patient is a list of networkx DiGraph objects.
        """

        # list of patients with networkx DiGraph objects
        dataset_nx = []

        # iterate through all patients and convert each TumorGraph into a networkx DiGraph
        for patient in self.dataset:
            patient_nx = []
            for graph in patient:
                patient_nx.append(graph.to_DiGraph())
            dataset_nx.append(patient_nx)
        
        return dataset_nx

    def save_dataset(self, dataset_path):
        """
        Saves the dataset in a .txt file with proper format.

        Parameters:
        - dataset_path: path to the .txt file where the dataset will be saved.
        """

        # open the file in write mode
        with open(dataset_path, "w") as file:

            # write the number of patients
            file.write(f'{self.n_patients()} patients\n')

            # iterate through patients and save their graphs
            for i, patient in enumerate(self.dataset):
                file.write(f'{len(patient)} graphs for patient {i}\n')
                for j, graph in enumerate(patient):
                    file.write(f'{graph.n_nodes()} nodes in graph {j}\n')
                    for node in graph.nodes:
                        file.write(f'{node.id} ')
                        labels = node.labels
                        file.write(f'{labels[0]}')
                        labels = labels[1:]
                        for label in labels:
                            file.write(f',{label}')
                        file.write('\n')
                    file.write(f'{graph.n_edges()} edges in graph {j}\n')
                    for edge in graph.edges:
                        file.write(f'{edge[0]} {edge[1]}\n')

class TorchTumorDataset(torch.utils.data.Dataset):
    """
    Dataset of TumorGraphs that can be organized into a torch_geometric.loader.DataLoader object.
    An item in the dataset is a TumorGraph.
    """

    def __init__(self, patients, node_encoding_type="mutation", known_labels_mapping=None):
        """
        Constructor.

        Parameters:
        - patients: TumorDataset object.
        - node_encoding_type: string that specifies how to encode nodes in the input graphs.
                              It can be either "mutation" or "clone".
                              If "mutation", then each node is encoded only based on its label.
                              If "clone", then each node is encoded based on the clone sublying it, that is, all mutations labelling ancestor nodes.
        - known_labels_mapping: dictionary that assigns a unique integer id to a set of node labels. Only the keys of the dictionary can be used as node labels. Their ids are used to encode the nodes.
                             If None, then node labels in patients will be used and a mapping will be created.
                             Moreover, labels in patients that do not appear in known_labels_mapping will be replaced with label "unknown".
                             This option is useful for testing on a TorchTumorDataset object based on a model trained on another TorchTumorDataset object.
        """

        # if a node label mapping is provided as input, then replace node labels not in the mapping with the label "unknown"
        if known_labels_mapping is not None:
            patients.replace_label_set(set(known_labels_mapping.keys()))
            self.node_labels_mapping = known_labels_mapping
        
        # else, create a mapping using the node labels in patients
        else:
            self.node_labels_mapping = self.map_node_labels(patients.node_labels())

        # transform the list of TumorGraphs into a list of torch_geometric.data.Data objects
        encoded_dataset = self.encode_dataset(patients, node_encoding_type)
        
        # flatten the dataset so to have a list of graphs
        self.dataset = self.flatten_dataset(encoded_dataset)

    def __len__(self):
        """
        Computes the number of graphs in the dataset.

        Returns:
        - len(self.dataset): number of graphs in the dataset.
        """

        return len(self.dataset)
    
    def __getitem__(self, i):
        """
        Returns the graph with index i in the dataset in the format required to feed a GNN, that is, a torch_geometric.data.Data object.

        Parameters:
        - i: integer of the graph to be returned.

        Returns:
        - self.dataset[i]: torch_geometric.data.Data object representing the graph with index i in the dataset.
        """

        return self.dataset[i]

    def encode_clones_graph(self, graph, base_encodings):
        """
        Encodes nodes in a TumorGraph based on the clone sublying them.
        More specifically, uses BFS to compute the encoding of each node in the input graph as linear combination of the endodings of its parent nodes plus its own label encoding.
        The encoding of a node is the average of the encodings of its parent nodes plus its own label encoding.
        The first encoded node is the root, encoded as a vector of all 0s except the first component, which is 1.
        The encodings of all other nodes are computed as descruibed above, using a BFS starting from the root.

        Parameters:
        - graph: TumorGraph object representing the graph to be encoded.
        - base_encodings: torch tensor of shape [n_nodes, n_components_encoding] with node encodings for nodes in the input graph, such that each node is encoded just based on its label

        Returns:
        - node_encodings: torch tensor of shape [n_nodes, n_components_encoding] with node encodings for nodes in the input graph, such that each node is encoded based on the clone sublying it.
        """

        # transform the input graph into a networkx DiGraph
        graph_nx = graph.to_DiGraph()

        # find the root node: by construction, it is the node with label "root" and in-degree = 0
        root = None
        for node in graph_nx.nodes:
            if graph_nx.in_degree(node) == 0 and graph_nx.nodes[node]["labels"] == ["root"]:
                root = node
                break

        # initialize the tensor with node encodings to a copy of the base encodings
        node_encodings = base_encodings.clone()

        # perform a BFS starting from the root to compute the encoding of each node
        queue = [root]
        while len(queue) > 0:
            v = queue.pop(0)
            parents_v = list(graph_nx.predecessors(v))
            for parent in parents_v:
                node_encodings[v] += node_encodings[parent] / len(parents_v)
            queue.extend(list(graph_nx.successors(v)))
        
        return node_encodings

    def encode_graph(self, graph, pos, node_encoding_type):
        """
        Transforms the input tumorGraph into a torch_geometric Data object.
        Two ways of encoding nodes are possible: "mutation" or "clone".
        If node_encoding_type = "mutation", then each node is encoded as a vector of size len(self.node_labels_mapping) - 1 as follows:
        - "root": vector of all 0s, except the first component, which is 1;
        - "unknown": vector of all 0s except the last component, which is 1;
        - "empty": vector of all 0s;
        - all other labels, which refer to mutations, are encoded as one-hot vectors with all 0s and a 1 at the index that univoquely identifies
          the label.
        If node_encoding_type = "mutation", then each node is encoded as a vector of size len(self.node_labels_mapping) - 1 that is the sum of the encodings of the labels in the list of labels of the node itself.
        If node_encoding_type = "clone", then each node is encoded as a vector of size len(self.node_labels_mapping) - 1 that is the sum of the encodings of the labels of nodes in the path from the root to the node itself.
        Basically, the encoding of a node represents the clone sublying it.
        
        Parameters:
        - graph: TumorGraph object representing the graph to be encoded.
        - pos: tuple (i, j) indicating that the encoded graph is in position j for patient i.
        - node_encoding_type: string that specifies how to encode nodes in the input graphs.
                              It can be either "mutation" or "clone".
                              If "mutation", then each node is encoded only based on its list of labels.
                              If "clone", then each node is encoded based on the clone sublying it, that is, including also all mutations labelling ancestor nodes.

        Returns:
        - geo_graph: torch_geometric Data object.
                     geo_graph.x is the matrix with node encodings for nodes in the input graph. It is a tensor of shape [n_nodes, n_components_encoding].
                     geo_graph.edge_index is the list of edges in the input graph. It is a tensor of shape [2, n_edges].
                     We also add attribute geo_graph.id, that is a tuple (i, j) indicating that the encoded graph is in position j for patient i.
        """

        # initialize the tensor with node encodings for the nodes in the graph
        node_encodings = torch.zeros((graph.n_nodes(), len(self.node_labels_mapping) - 1), dtype=torch.float)

        # remap the graph so to have consecutive node ids, if necessary
        remapped_graph = self.remap_node_ids(graph)

        # encode each node in the graph based on its list of labels
        for node in remapped_graph.nodes:
            for label in node.labels:    
                if label != "empty":
                    node_encodings[node.id][self.node_labels_mapping[label]] += 1
        
        # if a clone-level node encoding is required, then compute it
        if node_encoding_type == "clone":
            node_encodings = self.encode_clones_graph(remapped_graph, node_encodings)

        # tensor with the edge list with shape [2, n_edges]
        edge_list = torch.tensor(remapped_graph.edges, dtype=torch.long).t().contiguous()

        # create the Data object
        geo_graph = Data(x=node_encodings, edge_index=edge_list)

        # add an attribute with the indices identifying the graph in the dataset
        geo_graph.id = pos

        return geo_graph

    def encode_dataset(self, patients, node_encoding_type):
        """
        Transforms a TumorDataset object into a list patients, where a patient is a list of torch_geometric Data objects representing the graphs assigned to it.
        The produced list is aligned with patients, i.e., patients[i][j] and geo_patients[i][j] represent the same graph.
        
        Parameters:
        - patients: TumorDataset object.
        - node_encoding_type: string that specifies how to encode nodes in the input graphs.
                              It can be either "mutation" or "clone".

        Returns:
        - geo_patients: list patients, where a patient is a list of torch_geometric Data objects representing the graphs assigned to it.
                        geo_patients[i][j] is the torch_geometric Data object representing graph patients[i][j].
        """

        # dataset that will contain torch_geometric.data.Data objects
        geo_patients = []

        # fill the dataset by transforming TumorGraph objects into torch_geometric.data.Data objects
        for i, patient in enumerate(patients.dataset):
            geo_patients.append([])
            for j, graph in enumerate(patient):
                geo_patients[i].append(self.encode_graph(graph, (i, j), node_encoding_type))

        return geo_patients

    @staticmethod
    def map_node_labels(node_labels):
        """
        Maps each node label to a unique integer id that identifies it.
        In particular, each node label is mapped into an integer in [-1, len(node_labels) - 2] as follows:
        - "root": 0;
        - "unknown": len(node_labels) - 2, i.e., the largest id;
        - "empty": -1;
        - all other labels, which refer to mutations, are mapped into [1, len(node_labels) - 3]
        Notice that, by construction, TumorData.node_labels() always contains "root", "unknown" and "empty".
        The mapping is done in alphabetical order, so to allow for reproducibility.

        Parameters:
        - node_labels: set with node labels to be mapped.

        Returns:
        - mapped_labels: dictionary with node labels as keys and corresponding integer ids as values.
                         The mapping is done in alphabetical order, so to allow for reproducibility.
        """
        
        # initialize the dictionary representing the mapping
        mapped_labels = {}

        # convert the set of node labels into a list and sort it so to allow for reproducibility
        node_labels_list = [label for label in node_labels if label not in ["empty", "root", "unknown"]]
        node_labels_list.sort()

        # assign ids to "root" and "empty"
        mapped_labels["empty"] = -1
        mapped_labels["root"] = 0

        # id to be assigned to the next node label
        id = 1

        # assign subsequent ids to the remaining node labels
        for label in node_labels_list:
            mapped_labels[label] = id
            id += 1

        # assign the id to "unknown"
        mapped_labels["unknown"] = id

        return mapped_labels

    @staticmethod
    def flatten_dataset(dataset):
        """
        Flattens the input dataset so to have a single list with all graphs in the input dataset.
        It also adds an attribute index to each graph, so to keep information about where it is in the flattened dataset, useful when batching.

        Parameters:
        - dataset: list of lists of graphs to be flattened.

        Returns:
        - flattened_dataset: list of graphs.
                             It is the flattened version of the input dataset, where graphs are ordered the following way:
                             [dataset[0][0], dataset[0][1], ..., dataset[0][len(dataset[0]) - 1], dataset[1][0], ..., dataset[1][len(dataset[1]) - 1], ..., dataset[len(dataset) - 1][len(dataset[len(dataset) - 1]) - 1].
                             Each graph, has an additional attribute index, which is an integer indicating the position of the graph in the flattened list of graphs.
        """

        # initialize the flattened dataset
        flattened_dataset = []

        # index of the next graph to append
        next_idx = 0

        # fill the flattened dataset in the right order
        for patient in dataset:
            for graph in patient:
                curr_graph = copy.deepcopy(graph)
                curr_graph.index = next_idx
                next_idx += 1
                flattened_dataset.append(curr_graph)
        
        return flattened_dataset

    @staticmethod
    def remap_node_ids(graph):
        """
        Remaps node ids in the input TumorGraph such that they are consecutive integers starting from 0.
        This is needed by torch_geometric.data.Data objects.
        The mapping is applied both to graph.nodes and graph.edges consistently.

        Parameters:
        - graph: TumorGraph object.

        Returns:
        - remapped_graph: TumorGraph object with node ids remapped.
        """

        # node ids in graph sorted in ascending order
        sorted_ids = graph.get_node_ids(sort=True)

        # if the ids are already consecutive, then return a copy of the original graph
        if sorted_ids[-1] == len(sorted_ids) - 1:
            return copy.deepcopy(graph)

        # else, create the mapping to consecutive node ids
        ids_mapping = {id: i for i, id in enumerate(sorted_ids)}

        # remap nodes and edges according to the mapping
        remapped_graph = TumorGraph()
        for node in graph.nodes:
            remapped_graph.add_node(TumorNode(node_id=ids_mapping[node.id], node_labels=node.labels))
        for edge in graph.edges:
            remapped_graph.add_edge((ids_mapping[edge[0]], ids_mapping[edge[1]]))
            
        return remapped_graph

class TumorGraphGNN(torch.nn.Module):
    """
    GNN to compute embeddings for tumor graphs.
    """

    def __init__(self, n_node_labels, h_1, h_2, embedding_dim, dropout_prob_1=None, dropout_prob_2=None, batch_normalization=False, device=torch.device('cpu')):
        """
        Parameters:
        - n_node_labels: number of different node_labels. This is the size of the input vector representing a node.
        - h_1: dimension of the hidden node representations after the first graph convolutional layer.
        - h_2: dimension of the hidden node representations after the second graph convolutional layer.
        - embedding_dim: dimension of the final embedding for the overall graph, after the application of the final linear layer.
        - dropout_prob_1 is the dropout probability before the second graph convolutional layer. If None, dropout is not applied.
        - dropout_prob_2 is the dropout probability before the fully connected layer. If None, dropout is not applied.
        - batch_normalization: True if batch normalization has to be applied after each graph convolutional layer, False otherwise.
        - device: device where to move torch tensors.
        """

        # call the constructor of the parent class
        super(TumorGraphGNN, self).__init__()

        # number of node labels, i.e., input node size
        self.n_node_labels = n_node_labels
        
        # graph convolutional layers
        self.gconv_1 = GCNConv(self.n_node_labels - 1, h_1)   # -1 because the label "empty" is encoded as a vector of all zeros
        self.gconv_2 = GCNConv(h_1, h_2)

        # batch normalization layers
        self.use_batch_normalization = batch_normalization
        if self.use_batch_normalization:
            self.bn_1 = torch.nn.BatchNorm1d(h_1)
            self.bn_2 = torch.nn.BatchNorm1d(h_2)
        
        # dropout layers
        if dropout_prob_1 is None:
            self.dropout_1 = None
        else:
            self.dropout_1 = torch.nn.Dropout(dropout_prob_1)
        if dropout_prob_2 is None:
            self.dropout_2 = None
        else:
            self.dropout_2 = torch.nn.Dropout(dropout_prob_2)

        # fully connected layer
        self.fc = torch.nn.Linear(h_2, embedding_dim)

        # device on which to run the model
        self.device = device

        # move the model to the device
        self.to(self.device)

    def forward(self, V, E, batch_ids):
        """
        Performs the forward pass on a batch of graphs.

        Parameters:
        - V: tensor with node embeddings. The tensor has dimension 2. Each row corresponds to a node and represents
            its input embedding. Since the input is a batch of graphs, nodes of the same graph are stored as subsequent
            rows.
        - E: tensor with edges. The tensor has 2 dimensions. There is a column for each edge and two rows:
            the first row contains source nodes and the second one contains destination nodes.
            Edges belonging to the same graph are stored subsequently.
        - batch_ids: tensor of dimension 1 where batch_ids[i] specifies to which graph of those in the batch node V[i] belongs.
                    Moreover, edge j in E belongs to graph batch_ids[j].
                    The role of batch_ids is crucial, because graphs in the input batch in general have different dimensions.
        
        Returns:
        - G: final embedding representations for graphs in the input batch.
        """

        # move tensors to the device
        V = V.to(self.device)
        E = E.to(self.device)
        batch_ids = batch_ids.to(self.device)

        # first graph convolutional layer
        V = self.gconv_1(V, E)
        if self.use_batch_normalization:
            V = self.bn_1(V)
        V = F.relu(V)
        if self.dropout_1 is not None:
            V = self.dropout_1(V)

        # second graph convolutional layer
        V = self.gconv_2(V, E)
        if self.use_batch_normalization:
            V = self.bn_2(V)
        V = F.relu(V)

        # global mean pooling layer
        G = global_mean_pool(V, batch_ids)

        # fully connected layer
        if self.dropout_2 is not None:
            G = self.dropout_2(G)
        G = self.fc(G)

        return G

class TrainerTumorModel:
    """
    Class with static functions to train and evaluate a TumorGraphGNN.
    """

    @staticmethod
    def load_dataset_txt(path):
        """
        Loads the dataset saved in a .txt file in the format required by our model.

        Parameters:
        - path: path to the .txt file containing the dataset.

        Returns:
        - dataset: dictionary, where each entry is a patient, with the id as key and the list of TumorGraphs as value.
        """

        # initialize the dataset
        dataset = {}

        # open the file
        with open(path, 'r') as file:

            # the first word of the first line is the number of patients in the dataset
            n_patients = int(file.readline().split()[0])

            # iterate through patients to load them
            for i in range(n_patients):

                # the first line of a patient contains the number of graphs contained by the patient as first word and the patient id as last word
                line = file.readline().split()
                n_graphs = int(line[0])
                patient_id = line[-1]

                # initialize the patient
                dataset[patient_id] = []

                # iterate through the graphs for the current patient
                for j in range(n_graphs):
                    
                    # initialize the current TumorGraph for the current patient
                    curr_graph = TumorGraph()

                    # the first line related to a graph has the number of nodes in the graph as first word
                    n_nodes = int(file.readline().split()[0])

                    # iterate through the nodes so to collect them
                    for k in range(n_nodes):
                        node = file.readline().split()
                        node_id = int(node[0])
                        node_labels = node[1].split(",")
                        curr_graph.add_node(TumorNode(node_id=node_id, node_labels=node_labels))

                    # the next line has the number of edges in the graph as first word
                    n_edges = int(file.readline().split()[0])

                    # iterate through the edges so to collect them
                    for k in range(n_edges):
                        edge = file.readline().split()
                        curr_graph.add_edge((int(edge[0]), int(edge[1])))
                    
                    # append the current TumorGraph to the list of graphs for the current patient
                    dataset[patient_id].append(curr_graph)
            
        return dataset

    @staticmethod
    def get_dataloader(data, batch_size=16, shuffle=True):
        """
        Returns the torch_geometric DataLoader object with the input data.

        Parameters:
        - data: TorchTumorDataset to be put in the dataloader.
        - batch_size: batch size to be used for the dataloader.
        - shuffle: True if data has to shuffled, False otherwise.

        Returns:
        - dataloader: torch_geometric.loader.DataLoader object with the input data.
        """

        return GeoDataLoader(data, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def train_val_split(phylogenies, val_proportion=0.2, rd_seed=None):
        """
        Splits the input dataset into training and validation sets.

        Parameters:
        - phylogenies: dictionary with the dataset to be split.
        - val_proportion: float with the proportion of patients to be inserted in the validation set.
        - rd_seed: integer with the random seed for reproducibility. Used for all random operations.

        Returns:
        - train_phylogenies: dictionary with the training set.
        - val_phylogenies: dictionary with the validation set.
        """

        # set the random seed for reproducibility
        if rd_seed is not None:
            np.random.seed(rd_seed)

        # compute the ids
        ids = list(phylogenies.keys())

        # shuffle the ids
        np.random.shuffle(ids)

        # compute training and validation ids
        n_val = int(len(ids) * val_proportion)
        train_ids = ids[:-n_val]
        val_ids = ids[-n_val:]

        # create the training and validation sets
        train_phylogenies = {patient_id: phylogenies[patient_id] for patient_id in train_ids}
        val_phylogenies = {patient_id: phylogenies[patient_id] for patient_id in val_ids}

        return train_phylogenies, val_phylogenies

    @staticmethod
    def load_train_val_data(phylogenies_path, val_proportion=0.2, rd_seed=None, min_label_occurrences=0, node_encoding_type='clone', batch_size=64, device=torch.device('cpu')):
        """
        Loads the training and validation datasets with phylogenies.
        It also returns the number of labels present in the training set, necessary to define the input size of the TumorGraphGNN,
        and the distances between all pairs of graphs in the training and validation datasets.

        Parameters:
        - phylogenies_path: string with the path to the file with phylogenies.
        - val_proportion: float with the proportion of patients to be inserted in the validation set.
        - rd_seed: integer with the random seed for reproducibility. Used for all random operations.
        - min_label_occurrences: integer with the minimum number of occurrences for a label to be considered in the training set.
        - node_encoding_type: string with the type of encoding to be used for the nodes in the phylogenies. Can be either 'mutation' or 'clone'.
        - batch_size: batch size to be used for the dataloaders.
        - device: device to be used for the computations.

        Returns:
        - train_torch_data: TorchTumorDataset object with the training data.
        - val_torch_data: TorchTumorDataset object with the validation data.
        - train_distances: torch tensor with the distances between all pairs of graphs in the training dataset.
        - val_distances: torch tensor with the distances between all pairs of graphs in the validation dataset.
        - n_labels: number of labels present in the training set.
        """

        # load the dataset with phylogenies
        phylogenies = TrainerTumorModel.load_dataset_txt(phylogenies_path)

        # split the dataset into training and validation sets
        train_phylogenies, val_phylogenies = TrainerTumorModel.train_val_split(phylogenies, val_proportion=val_proportion, rd_seed=rd_seed)

        # create TumorDataset objects for training and validation sets that contains patients sorted by patient id so to allow for reproducibility
        train_sorted_keys = sorted(train_phylogenies.keys())
        train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
        train_data = TumorDataset(train_list_patients)

        val_sorted_keys = sorted(val_phylogenies.keys())
        val_list_patients = [val_phylogenies[key] for key in val_sorted_keys]
        val_data = TumorDataset(val_list_patients)

        # compute the set of labels to be considered, based on the number of occurrences in the training set
        if min_label_occurrences > 0:
            train_data.remove_infreq_labels(min_label_occurrences)
            val_data.replace_label_set(train_data.node_labels(), replace_with='empty')

        # sample one graph per patient
        train_data.sample_one_graph_per_patient(rd_seed=rd_seed)
        val_data.sample_one_graph_per_patient(rd_seed=rd_seed)

        # create the TorchTumorDataset objects for training and val sets
        train_torch_data = TorchTumorDataset(train_data, node_encoding_type=node_encoding_type)
        val_torch_data = TorchTumorDataset(val_data, node_encoding_type=node_encoding_type, known_labels_mapping=train_torch_data.node_labels_mapping)

        # compute the tensors with the distances between all pairs of graphs in the training and validation datasets
        train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)
        val_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(val_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

        # number of labels that appear in the training set
        n_labels = len(train_data.node_labels())

        return train_torch_data, val_torch_data, train_distances, val_distances, n_labels

    @staticmethod
    def load_train_data(phylogenies_path, rd_seed=None, min_label_occurrences=0, node_encoding_type='clone', batch_size=64, device=torch.device('cpu')):
        """
        Loads the training datasets with phylogenies.
        It also returns the number of labels present in the training set, necessary to define the input size of the TumorGraphGNN,
        and the distances between all pairs of graphs in the training dataset.

        Parameters:
        - phylogenies_path: string with the path to the file with phylogenies.
        - rd_seed: integer with the random seed for reproducibility. Used for all random operations.
        - min_label_occurrences: integer with the minimum number of occurrences for a label to be considered in the training set.
        - node_encoding_type: string with the type of encoding to be used for the nodes in the phylogenies. Can be either 'mutation' or 'clone'.
        - batch_size: batch size to be used for the dataloaders.
        - device: device to be used for the computations.

        Returns:
        - train_torch_data: TorchTumorDataset object with the training data.
        - train_distances: torch tensor with the distances between all pairs of graphs in the training dataset.
        - n_labels: number of labels present in the training set.
        """

        # load the dataset with phylogenies
        train_phylogenies = TrainerTumorModel.load_dataset_txt(phylogenies_path)

        # create TumorDataset object for the training set that contains patients sorted by patient id so to allow for reproducibility
        train_sorted_keys = sorted(train_phylogenies.keys())
        train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
        train_data = TumorDataset(train_list_patients)

        # compute the set of labels to be considered, based on the number of occurrences in the training set
        if min_label_occurrences > 0:
            train_data.remove_infreq_labels(min_label_occurrences)

        # sample one graph per patient
        train_data.sample_one_graph_per_patient(rd_seed=rd_seed)

        # create the TorchTumorDataset for the training set
        train_torch_data = TorchTumorDataset(train_data, node_encoding_type=node_encoding_type)

        # compute the tensor with the distances between all pairs of graphs in the training dataset
        train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

        # number of labels that appear in the training set
        n_labels = len(train_data.node_labels())

        return train_torch_data, train_distances, n_labels

    @staticmethod
    def get_embeddings(model, dataloader, device=torch.device('cpu')):
        """
        Extracts the embeddings from the input dataloader.
        Be aware that the node labels appearing in the dataset must be present in the dataset used to train the TumorGraphGNN.

        Parameters:
        - model: TumorGraphGNN object used to compute the embeddings.
        - data: DataLoader object with the graphs to be embedded. The dataloader must contain graphs sorted by patient id.
        - device: device to be used for the computations.

        Returns:
        - embeddings: torch tensor with the embeddings of the graphs in the input data.
                      embeddings and data are aligned, meaning that embeddings[i] is the embedding of the graph data[i].
        """

        # set the model to evaluation mode
        model.eval()

        # compute embeddings for all graphs
        embeddings = {}
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                batch_embeddings = model(batch.x, batch.edge_index, batch.batch)
                for i in range(len(batch)):
                    embeddings[batch[i].index] = batch_embeddings[i]
        
        # convert embeddings into a tensor
        sorted_indices = sorted(embeddings.keys())
        embeddings = torch.stack([embeddings[i] for i in sorted_indices], dim=0).to(device)
        
        return embeddings

    @staticmethod
    def compute_absolute_errors(embeddings, graph_distances, device=torch.device('cpu')):
        """
        Computes the absolute errors between Euclidean distance of embeddings and true graph distance between corresponding graphs.
        For each pair of graphs, the absolute error between their true graph distance and the Euclidean distance of their embeddings is computed.

        Parameters:
        - embeddings: torch tensor of shape (n_graphs, embedding_size) with embeddings.
        - graph_distances: torch tensor of shape (n_graphs, n_graphs) with the true pairwise graph distances.
        - device: device to be used for the computations.

        Returns:
        - torch.abs(emb_dts - graph_distances): torch tensor of shape (n_graphs, n_graphs) where errors[i][j] = |norm(embeddings[i] - embeddings[j]) - graph_distances[i][j]|
                                                The matrix is symmetric, meaning that errors[i][j] = errors[j][i].
        """

        # move tensors to self.device
        embeddings = embeddings.to(device)
        graph_distances = graph_distances.to(device)

        # use broadcasting so to compute pairwise embeddings distances
        emb_diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)      # shape (n_graphs, n_graphs, embedding_size)
        emb_dist = torch.linalg.vector_norm(emb_diff, dim=-1)             # shape (n_graphs, n_graphs)

        # compute errors as the absolute value between graph distance and embedding distance and return them
        return torch.abs(emb_dist - graph_distances)

    @staticmethod
    def compute_squared_errors(embeddings, graph_distances, device=torch.device('cpu')):
        """
        Computes the squared errors between Euclidean distance of embeddings and true graph distance between corresponding graphs.
        For each pair of graphs, the squared error between their true graph distance and the Euclidean distance of their embeddings is computed.

        Parameters:
        - embeddings: torch tensor of shape (n_graphs, embedding_size) with embeddings.
        - graph_distances: torch tensor of shape (n_graphs, n_graphs) with the true pairwise graph distances.
        - device: device to be used for the computations.

        Returns:
        - (emb_dists - graph_distances) ** 2: torch tensor of shape (n_graphs, n_graphs) where errors[i][j] = (norm(embeddings[i] - embeddings[j]) - graph_distances[i][j]) ** 2
                                            The matrix is symmetric, meaning that errors[i][j] = errors[j][i].
        """

        # move tensors to self.device
        embeddings = embeddings.to(device)
        graph_distances = graph_distances.to(device)

        # use broadcasting so to compute pairwise embeddings distances
        emb_diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)      # shape (n_graphs, n_graphs, embedding_size)
        emb_dist = torch.linalg.vector_norm(emb_diff, dim=-1)             # shape (n_graphs, n_graphs)

        # compute errors as the squared difference between graph distance and embedding distance and return them
        return (emb_dist - graph_distances) ** 2

    @staticmethod
    def compute_mean_absolute_error(embeddings, graph_distances, device=torch.device('cpu')):
        """
        Computes the mean absolute error between Euclidean distance of embeddings and true graph distance between corresponding graphs.
        The mean absolute error is computed as the average of the absolute errors between the true graph distances and the Euclidean distances of the embeddings.

        Parameters:
        - embeddings: torch tensor of shape (n_graphs, embedding_size) with embeddings.
        - graph_distances: torch tensor of shape (n_graphs, n_graphs) with the true pairwise graph distances.
        - device: device to be used for the computations.

        Returns:
        - torch.mean(errors): mean absolute error between the Euclidean distance of embeddings and the true graph distances.
        """

        # move tensors to self.device
        embeddings = embeddings.to(device)
        graph_distances = graph_distances.to(device)

        # find the indices of the upper triangular matrix so not to count distances twice and exclude the main diagonal
        triu_indices = torch.triu_indices(graph_distances.shape[0], graph_distances.shape[1], offset=1) # shape (2, n_pairs = batch_size * (batch_size - 1) // 2), where the first row contains row indices and the second row contains column indices

        # compute the mean absolute error
        errors = TrainerTumorModel.compute_absolute_errors(embeddings, graph_distances)
        errors = errors[triu_indices[0], triu_indices[1]]
        
        # compute the mean absolute error and return it
        return torch.mean(errors)

    @staticmethod
    def compute_mean_squared_error(embeddings, graph_distances, device=torch.device('cpu')):
        """
        Computes the mean squared error between Euclidean distance of embeddings and true graph distance between corresponding graphs.
        The mean squared error is computed as the average of the squared errors between the true graph distances and the Euclidean distances of the embeddings.

        Parameters:
        - embeddings: torch tensor of shape (n_graphs, embedding_size) with embeddings.
        - graph_distances: torch tensor of shape (n_graphs, n_graphs) with the true pairwise graph distances.
        - device: device to be used for the computations.

        Returns:
        - torch.mean(errors): mean squared error between the Euclidean distance of embeddings and the true graph distances.
        """

        # move tensors to self.device
        embeddings = embeddings.to(device)
        graph_distances = graph_distances.to(device)

        # find the indices of the upper triangular matrix so not to count distances twice and exclude the main diagonal
        triu_indices = torch.triu_indices(graph_distances.shape[0], graph_distances.shape[1], offset=1)

        # compute the mean squared error
        errors = TrainerTumorModel.compute_squared_errors(embeddings, graph_distances)
        errors = errors[triu_indices[0], triu_indices[1]]

        # compute the mean squared error and return it
        return torch.mean(errors)

    @staticmethod
    def compute_R2(embeddings, graph_distances, device=torch.device('cpu')):
        """
        Computes the R^2 score between the Euclidean distance of embeddings and the true graph distance between corresponding graphs.
        The R^2 score is computed as 1 - SS_res / SS_tot, where SS_res is the sum of squared residuals and SS_tot is the total sum of squares.

        Parameters:
        - embeddings: torch tensor of shape (n_graphs, embedding_size) with embeddings.
        - graph_distances: torch tensor of shape (n_graphs, n_graphs) with the true pairwise graph distances.
        - device: device to be used for the computations.

        Returns:
        - 1 - SS_res / SS_tot: R^2 score between the Euclidean distance of embeddings and the true graph distances.
        """

        # move tensors to self.device
        embeddings = embeddings.to(device)
        graph_distances = graph_distances.to(device)

        # find the indices of the upper triangular matrix so not to count distances twice and exclude the main diagonal
        triu_indices = torch.triu_indices(graph_distances.shape[0], graph_distances.shape[1], offset=1) # shape (2, n_pairs = batch_size * (batch_size - 1) // 2), where the first row contains row indices and the second row contains column indices

        # compute the SS_res
        SS_res = TrainerTumorModel.compute_squared_errors(embeddings, graph_distances)
        SS_res = torch.sum(SS_res)

        # extract true distances without duplicates and main diagonal entries
        true_distances = graph_distances[triu_indices[0], triu_indices[1]]

        # compute the average of the true distances
        avg_true_distances = torch.mean(true_distances)

        # compute the SS_tot
        SS_tot = torch.sum((true_distances - avg_true_distances) ** 2)

        # compute the R^2 score and return it
        return 1 - SS_res / SS_tot

    @staticmethod
    def compute_loss(embeddings, graph_distances, loss_fn=torch.nn.MSELoss, device=torch.device('cpu')):
        """
        Computes the loss of the input embeddings with respect to their true distances using the input loss function.

        Parameters:
        - embeddings: torch tensor of shape (n_graphs, embedding_size) with embeddings.
        - graph_distances: torch tensor of shape (n_graphs, n_graphs) with the true pairwise graph distances.
        - loss_fn: loss function to be applied.
        - device: device to be used for the computations.

        Returns:
        - loss_fn(emb_dist, true_dist): torch tensor of shape (1,) as a result of the application of loss_fn to the pairwise embedding distances and pairwise graph distances.
        """

        # move tensors to self.device
        embeddings = embeddings.to(device)
        graph_distances = graph_distances.to(device)

        # use broadcasting so to compute pairwise embeddings distances
        emb_diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)      # shape (n_graphs, n_graphs, embedding_size)
        emb_dist = torch.linalg.vector_norm(emb_diff, dim=-1)             # shape (n_graphs, n_graphs)

        # compute the indices of the upper triangular matrix so not to count distances twice and exclude the main diagonal
        triu_indices = torch.triu_indices(emb_diff.shape[0], emb_diff.shape[1], offset=1) # shape (2, n_pairs = batch_size * (batch_size - 1) // 2), where the first row contains row indices and the second row contains column indices

        # extract embedding distances and true distances according to the computed indices
        emb_dist = emb_dist[triu_indices[0], triu_indices[1]]
        true_dist = graph_distances[triu_indices[0], triu_indices[1]]

        # return the application of the input loss function to embedding distances and true graph distances
        return loss_fn(emb_dist, true_dist)

    @staticmethod
    def train_epoch(model, train_dataloader, train_graph_distances, loss_fn=torch.nn.MSELoss, opt=torch.optim.Adam, val_dataloader=None, val_graph_distances=None, device=torch.device('cpu')):
        """
        Trains a TumorGraphGNN on an input training data for a single epoch.
        A validation set can be passed as input as well, in which case it is used for evaluation.

        Parameters:
        - model: TumorGraphGNN object to be trained.
        - train_dataloader: torch_geometric DataLoader with training data.
        - train_graph_distances: 2D tensor with pairwise distances for the graphs in the training dataloader.
        - loss_fn: loss function to be used for training.
        - opt: optimizer to be used to train the tumorTreeGNN.
        - val_dataloader: torch_geometric DataLoader with validation data. If None, no validation is performed.
        - val_graph_distances: 2D tensor with pairwise distances for the graphs in the validation dataloader. If None, no validation is performed.
        - device: device to be used for the computations.

        Returns:
        - avg_epoch_loss: loss computed for training data during the current epoch, averaged across batches.
        - val_avg_loss: average loss computed for validation data.
                        None is returned if validation data is provided as input.
        """

        # flag indicating whether a validation set is provided as input
        val_flag = True
        if val_dataloader is None or val_graph_distances is None:
            val_flag = False

        # set the model to train mode
        model.train()
        
        # set loss for the current epoch to 0
        epoch_loss = 0
        
        # iterate through batches
        for batch in train_dataloader:

            # move the batch to the device
            batch = batch.to(device)
            
            # reset gradients computed for the previous batch to 0
            opt.zero_grad()

            # forward pass to get embeddings for all graphs in the batch
            embeddings = model(batch.x, batch.edge_index, batch.batch)

            # number of graphs in the batch
            batch_size = batch.num_graphs
            
            # add new dimensions to embeddings, which has shape (batch_size, embed_dim), so to be able to broadcast it then
            embeds_0 = embeddings.unsqueeze(0)                                                  # shape (1, batch_size, embed_dim)
            embeds_1 = embeddings.unsqueeze(1)                                                  # shape (batch_size, 1, embed_dim)

            # compute the norm of the difference for each pair of embeddings
            embeds_diff = embeds_0 - embeds_1                                                   # shape (batch_size, batch_size, embed_dim)
            emb_distances = torch.linalg.vector_norm(embeds_diff, dim=-1)                       # norm in the embed_dim dimension, so shape (batch_size, batch_size)

            # compute the indices of the upper triangular matrix right above the main diagonal, so to exclude differences computed twice and of embeddings with themselves
            triu_indices = torch.triu_indices(batch_size, batch_size, offset=1)                 # shape (2, n_pairs = batch_size * (batch_size - 1) // 2), where the first row contains row indices and the second row contains column indices

            # use the computed indices to extract the tensor with embedding distances
            emb_distances = emb_distances[triu_indices[0], triu_indices[1]]                     # shape (n_pairs)

            # extract the true distances using the attribute index of each graph in the batch
            true_distances = []
            for i, j in zip(triu_indices[0], triu_indices[1]):
                true_distances.append(train_graph_distances[batch[i].index, batch[j].index])
            true_distances = torch.stack(true_distances).squeeze().to(device)                  # shape (n_pairs)
            
            # compute the loss and backpropagate
            loss = loss_fn(emb_distances, true_distances)
            loss.backward()
            opt.step()

            # add the loss for the current batch to the current value of the loss for the current epoch
            epoch_loss += loss.item()

        # average the epoch loss over the number of batches
        avg_epoch_loss = epoch_loss / len(train_dataloader)

        # evaluate the model on the validation set, if needed
        val_avg_loss = None
        if val_flag:
            model.eval()
            val_embeds = TrainerTumorModel.get_embeddings(model, val_dataloader, device=device)
            val_avg_loss = TrainerTumorModel.compute_loss(val_embeds, val_graph_distances, loss_fn=loss_fn, device=device).item()

        return avg_epoch_loss, val_avg_loss

    @staticmethod
    def train(
            model,
            train_data,
            train_graph_distances,
            loss_fn=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            weight_decay=0,
            batch_size=16,
            val_data=None,
            val_graph_distances=None,
            plot_save=None,
            verbose=True,
            epochs=100,
            lr=0.001,
            early_stopping_tolerance=None,
            save_model=None,
            device=torch.device('cpu')
            ):
        """
        Trains a TumorGraphGNN on an input training data, printing and plotting information, if required.
        A validation set can be passed as input as well, in which case it is included in the printed and plotted information.

        Parameters:
        - model: TumorGraphGNN object to be trained.
        - train_data: TorchTumorDataset object with training data.
        - train_graph_distances: 2D tensor with pairwise distances for the graphs in the training dataloader.
        - loss_fn: loss function to be used for training.
        - optimizer: optimizer to be used to train the tumorGraphGNN.
        - weight_decay: weight decay to be used for the optimizer.
        - batch_size: batch size.
        - val_data: TorchTumorDataset object with validation data. If None, no validation is performed.
        - val_graph_distances: 2D tensor with pairwise distances for the graphs in the validation dataloader. If None, no validation is performed.
        - plot_save: path where to save the plot with losses. If None, then no plot is created.
        - verbose: True if training information must be print, False otherwise.
        - epochs: number of training iterations.
        - lr: learning rate.
        - early_stopping_tolerance: number of epochs without improvement after which the training process is stopped. If None, early stopping is not applied.
        - save_model: path where to save the model. If None, then no model is saved.
        - device: device to be used for the computations.
        """

        # flag indicating whether a validation set is provided as input
        val_flag = True
        if val_data is None or val_graph_distances is None:
            val_flag = False
        
        # create DataLoaders
        train_dl = TrainerTumorModel.get_dataloader(train_data, batch_size=batch_size)
        if val_flag:
            val_dl = TrainerTumorModel.get_dataloader(val_data, batch_size=batch_size, shuffle=False)
        else:
            val_dl = None

        # initialize early stopping variables, if enabled
        if val_flag and early_stopping_tolerance is not None:
            best_val_loss = float('inf')
            early_stop_counter = 0

        # initialize the optimizer
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

        # print that training is starting
        print('\nTraining:\n')

        # initialize the lists of losses computed at each epoch
        train_losses_epochs_list = []
        if val_flag:
            val_losses_epochs_list = []
    
        # iterate through epochs
        for epoch in range(epochs):

            # train the model and compute losses
            train_loss, val_loss = TrainerTumorModel.train_epoch(
                model,
                train_dl,
                train_graph_distances,
                loss_fn=loss_fn,
                opt=opt,
                val_dataloader=val_dl,
                val_graph_distances=val_graph_distances,
                device=device
            )

            # append losses to the lists
            train_losses_epochs_list.append(train_loss)
            if val_loss is not None:
                val_losses_epochs_list.append(val_loss)

                # if early stopping is enabled, then check if the training process needs to be stopped
                if early_stopping_tolerance is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    if early_stop_counter >= early_stopping_tolerance:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

            # print the loss values for the current epoch, if required
            if verbose:
                if val_loss is not None:
                    print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss} | Validation Loss: {val_loss}")
                else:
                    print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss}")

        # save a plot with losses over time, if required
        if plot_save is not None:
            if val_flag:
                TrainerTumorModel.plot_losses([train_losses_epochs_list, val_losses_epochs_list], plot_save, ['Train', 'Validation'], 'Losses over Epochs')
            else:
                TrainerTumorModel.plot_losses([train_losses_epochs_list], plot_save, ['Train'], 'Loss over Epochs')
        
        # save the trained model, if required
        if save_model is not None:
            os.makedirs(os.path.dirname(save_model), exist_ok=True)
            torch.save(model.state_dict(), save_model)
    
    @staticmethod
    def tuning_objective(trial, train_torch_data, val_torch_data, train_distances, val_distances, n_labels, random_seed=None, device=torch.device('cpu')):
        """
        Objective function to be used in the hyperparameter tuning process.

        Parameters:
        - trial: optuna.Trial object to be used for hyperparameter tuning.
        - train_torch_data: TorchTumorDataset object with training data.
        - val_torch_data: TorchTumorDataset object with validation data.
        - train_distances: 2D tensor with pairwise distances for the graphs in the training dataset.
        - val_distances: 2D tensor with pairwise distances for the graphs in the validation dataset.
        - n_labels: number of labels present in the training set.
        - random_seed: integer with the random seed for reproducibility. Used for all random operations.
        - device: device to be used for tensor operations.

        Returns:
        - censored_c_index: censored c-index computed by the trained model on the validation set.
        """

        # suggest hyperparameters
        weight_decay = trial.suggest_float('weight_decay', 1e-06, 1e-01, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        loss_fn = trial.suggest_categorical('loss_fn', ['MAE_loss'])
        optimizer = trial.suggest_categorical('optimizer', ['Adam'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        epochs = trial.suggest_categorical('epochs', [30, 50, 100, 200, 300])
        batch_normalization = trial.suggest_categorical('batch_normalization', [True, False])
        dropout_prob_1 = trial.suggest_float('dropout_prob_1', 0.0, 0.8, step=0.1)
        dropout_prob_2 = trial.suggest_float('dropout_prob_2', 0.0, 0.8, step=0.1)
        h_1 = trial.suggest_categorical('h_1', [16, 32, 64, 128])
        h_2 = trial.suggest_categorical('h_2', [16, 32, 64, 128])
        embedding_dim = trial.suggest_categorical('embedding_dim', [16, 32, 64, 128])

        # create a TumorGraphGNN instance with input size based on the labels in the dataset
        model = TumorGraphGNN(
            n_node_labels=n_labels,
            h_1=h_1,
            h_2=h_2,
            embedding_dim=embedding_dim,
            dropout_prob_1=dropout_prob_1,
            dropout_prob_2=dropout_prob_2,
            batch_normalization=batch_normalization,
            device=device
        )

        # create the training and validation dataloaders
        train_dataloader = TrainerTumorModel.get_dataloader(train_torch_data, batch_size=batch_size)
        val_dataloader = TrainerTumorModel.get_dataloader(val_torch_data, batch_size=batch_size, shuffle=False)

        # initialize the optimizer
        optimizer = Utils.select_optimizer(optimizer)
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

        # initialize the loss function
        loss_fn = Utils.select_loss(loss_fn)
    
        # iterate through epochs
        for epoch in range(epochs):

            # train the model and compute losses
            train_loss, val_loss = TrainerTumorModel.train_epoch(
                model,
                train_dataloader,
                train_graph_distances=train_distances,
                loss_fn=loss_fn,
                opt=opt,
                val_dataloader=val_dataloader,
                val_graph_distances=val_distances,
                device=device
            )

            # report the validation loss to optuna
            trial.report(val_loss, epoch)

            # check if the training process needs to be stopped because the trial is not promising
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        # compute the loss for the validation set
        val_embeddings = TrainerTumorModel.get_embeddings(model, val_dataloader, device=device)
        val_loss = TrainerTumorModel.compute_loss(val_embeddings, val_distances, loss_fn=loss_fn, device=device).item()

        return val_loss

    @staticmethod
    def plot_losses(losses_lists, save_path, lists_labels, plot_title):
        """
        Saves a plot showing the input lists of losses.
        The x axis shows epochs and in the y axis it is possible to see the value of the loss.
        A curve is created for each list of losses in losses_lists and all curves are plotted in the same figure.

        Parameters:
        - losses_lists: list of lists of losses. Lists must be aligned, i.e., losses_lists[i][k] and losses_lists[j][k] must be the losses in two different sets after epoch k.
                        For instance, losses[i] can store training losses over epochs and losses[j] can store validation losses over epochs.
        - save_path: path where to save the plot.
        - lists_labels: list of strings with the labels for the curves to be plotted. labels[i] is the label for losses_lists[i].
        - plot_title: title of the plot.
        """

        # set the theme
        sns.set_theme('paper')

        # plot a curve for each list of losses
        for i, losses in enumerate(losses_lists):
            plt.plot(losses, label=lists_labels[i])

        # set some plot properties
        plt.title(plot_title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # save the plot
        plt.savefig(save_path)

class GraphDistances:
    """
    Class with static methods to compute distance between graphs.
    """

    @staticmethod
    def compute_ancestry_set(graph):
        """
        Computes the set of all ancestor-descendant pairs of node labels in the input graph.
        
        Parameters:
        - graph: NetworkX DiGraph representing a directed graph.

        Returns:
        - AD_pairs: ancestor-descendant pairs of node labels in the input graph.
        """

        # AD pairs in the input graph
        AD_pairs = set()

        # iterate over nodes in the graph
        for node in graph.nodes:
            ancestors = nx.ancestors(graph, node)
            for anc in ancestors:
                for anc_label in graph.nodes[anc]['labels']:
                    for node_label in graph.nodes[node]['labels']:
                        AD_pairs.add((anc_label, node_label))
        
        return AD_pairs

    @staticmethod
    def ancestor_descendant_dist(graph_1, graph_2):
        """
        Computes the ancestor-descendant distance between the two input graphs.

        Parameters:
        - graph_1: networkx DiGraph.
        - graph_2: networkx DiGraph.
            
        Returns:
        - len(symmetric_diff): AD distance between the two input graphs.
        """

        # ancestry sets of the two graphs, i.e., the sets with all ancestor-descendant pairs in each graph
        A_1 = GraphDistances.compute_ancestry_set(graph_1)
        A_2 = GraphDistances.compute_ancestry_set(graph_2)

        # compute the symmetric difference between A_1 and A_2
        symmetric_diff = A_1.symmetric_difference(A_2)

        # return the number of ancestor-descendant pairs in the symmetric_diff set
        return len(symmetric_diff)

    @staticmethod
    def compute_distances(graphs, graph_dist_fn):
        """
        Computes all distances between graphs in the input dataset, using the input graph distance function.

        Parameters:
        - graphs: list of graphs. Each graph is networkx DiGraph.
        - graph_dist_fn: graph distance function to be used to compute the distance between two graphs.

        Returns:
        - distances: torch tensor of shape (len(graphs), len(graphs)) with the distances between all pairs of graphs.
        """

        # number of graphs in the input dataset
        n_graphs = len(graphs)

        # initialize the tensor to store the distances between all pairs of graphs
        distances = torch.zeros((n_graphs, n_graphs))

        # iterate over all pairs of graphs to compute distances
        for i in tqdm(range(n_graphs), desc='Computing distances', unit='graphs'):
            for j in range(i, n_graphs):
                dist = graph_dist_fn(graphs[i], graphs[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances
