import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from tumor_model import TumorGraphGNN
from tumor_model import TorchTumorDataset
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import utils as Utils
from tumor_model import TumorNode
from tumor_model import TumorGraph
from tumor_model import TumorDataset
from tumor_model import TrainerTumorModel
from sksurv.metrics import concordance_index_censored
import torch.utils
import torch.utils.data
from torch_geometric.data import Data
import copy

class TorchSurvivalDataset(torch.utils.data.Dataset):
    """
    Dataset with the TumorGraph and clinical survival label for each patient that can be organized into a torch dataloader.
    An item in the dataset is a tuple (phylogeny, survival_time, survival_event) for a patient.
    """

    def __init__(self, phylogenies, survival_data, node_encoding_type="mutation", known_labels_mapping=None):
        """
        Constructor.

        Parameters:
        - phylogenies: TumorDataset object with a phylogeny for each patient.
        - survival_data: array of arrays [times, events] with the survival data for each patient.
                         survival_data[0][i] is the survival time for patient with phylogeny phylogenies[i].
                         survival_data[1][i] is the survival event for patient with phylogeny phylogenies[i].
        - node_encoding_type: string that specifies how to encode nodes in the input graphs.
                              It can be either "mutation" or "clone".
                              If "mutation", then each node is encoded only based on its label.
                              If "clone", then each node is encoded based on the clone sublying it, that is, all mutations labelling ancestor nodes.
        - known_labels_mapping: dictionary that assigns a unique integer id to a set of node labels. Only the keys of the dictionary can be used as node labels. Their ids are used to encode the nodes.
                             If None, then node labels in phylogenies will be used and a mapping will be created.
                             Moreover, labels in phylogenies that do not appear in known_labels_mapping will be replaced with label "unknown".
                             This option is useful for testing on a TorchTumorDataset object based on a model trained on another TorchTumorDataset object.
        """

        # if a node label mapping is provided as input, then replace node labels not in the mapping with the label "unknown"
        if known_labels_mapping is not None:
            phylogenies.replace_label_set(set(known_labels_mapping.keys()))
            self.node_labels_mapping = known_labels_mapping
        
        # else, create a mapping using the node labels in phylogenies
        else:
            self.node_labels_mapping = self.map_node_labels(phylogenies.node_labels())

        # transform the list of TumorGraphs into a list of torch_geometric.data.Data objects
        encoded_phylogenies = self.encode_dataset(phylogenies, node_encoding_type)
        
        # flatten the dataset so to have a list of graphs
        encoded_phylogenies = self.flatten_dataset(encoded_phylogenies)

        # create the dataset as a list of tuples containing the encoded graph, survival time and survival event
        self.dataset = [(encoded_phylogenies[i], survival_data[0][i], survival_data[1][i]) for i in range(len(encoded_phylogenies))]

    def __len__(self):
        """
        Computes the number of tuples in the dataset.

        Returns:
        - len(self.dataset): number of tuples in the dataset.
        """

        return len(self.dataset)
    
    def __getitem__(self, i):
        """
        Returns the tuple with index i in the dataset.

        Parameters:
        - i: integer of the tuple to be returned.

        Returns:
        - self.dataset[i]: tuple with index i in the dataset. The tuple contains the encoded graph, survival time and survival event for patient i.
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

class SurvivalGNN(torch.nn.Module):
    """
    GNN to predict survival time.
    """

    def __init__(self, n_node_labels, h_1_dim, h_2_dim, hidden_dim, dropout_prob_1=None, dropout_prob_2=None, dropout_prob_3=None, batch_normalization=False, device=torch.device('cpu')):
        """
        Parameters:
        - n_node_labels: number of different node_labels. This is the size of the input vector representing a node.
        - h_1_dim: dimension of the hidden node representations after the first graph convolutional layer.
        - h_2_dim: dimension of the hidden node representations after the second graph convolutional layer.
        - hidden_dim: dimension of the hidden embedding for the overall graph, after the application of the first fully connected layer.
        - dropout_prob_1 is the dropout probability before the second graph convolutional layer. If None, dropout is not applied.
        - dropout_prob_2 is the dropout probability before the first fully connected layer. If None, dropout is not applied.
        - dropout_probs_3 is the dropout probability before the second fully connected layer. If None, dropout is not applied.
        - batch_normalization: True if batch normalization has to be applied after each graph convolutional layer, False otherwise.
        - device: device where to move torch tensors.
        """

        # call the constructor of the parent class
        super(SurvivalGNN, self).__init__()

        # number of node labels, i.e., input node size
        self.n_node_labels = n_node_labels
        
        # graph convolutional layers
        self.gconv_1 = GCNConv(self.n_node_labels - 1, h_1_dim)   # -1 because the label "empty" is encoded as a vector of all zeros
        self.gconv_2 = GCNConv(h_1_dim, h_2_dim)

        # batch normalization layers
        self.use_batch_normalization = batch_normalization
        if self.use_batch_normalization:
            self.bn_1 = torch.nn.BatchNorm1d(h_1_dim)
            self.bn_2 = torch.nn.BatchNorm1d(h_2_dim)
            self.bn_3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # dropout layers
        if dropout_prob_1 is None:
            self.dropout_1 = None
        else:
            self.dropout_1 = torch.nn.Dropout(dropout_prob_1)
        if dropout_prob_2 is None:
            self.dropout_2 = None
        else:
            self.dropout_2 = torch.nn.Dropout(dropout_prob_2)
        if dropout_prob_3 is None:
            self.dropout_3 = None
        else:
            self.dropout_3 = torch.nn.Dropout(dropout_prob_3)

        # fully connected layers
        self.fc_1 = torch.nn.Linear(h_2_dim, hidden_dim)
        self.fc_2 = torch.nn.Linear(hidden_dim, 1)

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
        - T: predicted survival times for the graphs in the input batch.
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

        # first fully connected layer
        if self.dropout_2 is not None:
            G = self.dropout_2(G)
        G = self.fc_1(G)
        if self.use_batch_normalization:
            G = self.bn_3(G)
        G = F.relu(G)

        # second fully connected layer
        if self.dropout_3 is not None:
            G = self.dropout_3(G)
        T = self.fc_2(G)

        return T

class SquaredMarginRankingLoss(torch.nn.Module):
    """
    Class implementing the squared version of the MarginRankingLoss function in PyTorch.
    """

    def __init__(self, margin=0.0, reduction='mean'):
        """
        Constructor.
        
        Parameters:
        - margin: margin to be used in the loss function.
        - reduction: reduction to be applied to the loss. Default is 'mean'.
        """

        # constructor of the parent class
        super(SquaredMarginRankingLoss, self).__init__()

        # set the margin
        self.margin = margin

        # set the reduction
        self.reduction = reduction

        # set the standard MarginRankingLoss function
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, x_1, x_2, y):
        """
        Forward pass of the loss function.

        Parameters:
        - x_1: tensor of shape (n_samples,) with input values.
        - x_2: tensor of shape (n_samples,) with input values.
        - y: tensor of shape (n_samples,) with values 1 or -1, indicating whether x_1 > x_2 or x_1 < x_2, respectively.

        Returns:
        - loss: loss computed for the input values. 
        """

        # compute the standard MarginRankingLoss
        loss = self.margin_ranking_loss(x_1, x_2, y)

        # square the loss
        loss **= 2

        # apply the reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss      

class TrainerSurvival:
    """
    Class with functions to train and evaluate a SurvivalGNN.
    """

    @staticmethod
    def select_loss_survival(string_loss):
        """
        Returns the loss function identified by the input string.

        Parameters:
        - string_loss: string with the name of the loss function to use.

        Returns:
        - loss function identified by the input string.
        """

        if string_loss == 'MarginRankingLoss':
            return torch.nn.MarginRankingLoss
        elif string_loss == 'SquaredMarginRankingLoss':
            return SquaredMarginRankingLoss
        
        raise ValueError('Invalid loss function name.')

    @staticmethod
    def custom_collate(batch):
        """
        Custom collate function to create a batch of items from a torch dataset.

        Parameters:
        - batch: list of items to be batched.

        Returns:
        - batch: batch of items.
        """

        # separate the three elements of each item in the batch
        graphs = [item[0] for item in batch]
        survival_times = [item[1] for item in batch]
        survival_events = [item[2] for item in batch]
        
        # batch the torch_geometric.data.Data objects
        batched_graphs = Batch.from_data_list(graphs)

        # return the batch as three elements
        return batched_graphs, torch.tensor(survival_times), torch.tensor(survival_events)

    @staticmethod
    def get_dataloader(data, batch_size=16, shuffle=True):
        """
        Returns the DataLoader object with the input data.

        Parameters:
        - data: TorchSurvivalDataset to be put in the dataloader.
        - batch_size: batch size to be used for the dataloader.
        - shuffle: True if data has to shuffled, False otherwise.


        Returns:
        - dataloader: DataLoader object with the input data.
        """

        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=TrainerSurvival.custom_collate)

    @staticmethod
    def comparable_mask(survival_times, survival_events):
        """
        Returns a mask to extract only comparable pairs of survival times from a torch tensor with all possible pairs of survival times.
        The set of comparable pairs is defined as the set of pairs of samples (i, j) such that survival_times[i] > survival_times[j] and survival_events[j] == 1.

        Parameters:
        - survival_times: tensor with survival times. Shape: (n_samples,).
        - survival_events: tensor with survival events. Shape: (n_samples,).

        Returns:
        - comp_mask: tensor with a mask for the comparable pairs. Shape: (n_samples, n_samples).
                     comp_mask[i, j] = 1 if the pair (survival_times[i], survival_time[j]) is comparable, 0 otherwise.

        """

        # compute the differences of all possible pairs of survival times using broadcasting
        times_diff = survival_times.unsqueeze(1) - survival_times.unsqueeze(0)                  # shape: (n_samples, n_samples)

        # create a mask to filter out pairs of samples that are not comparable
        times_mask = times_diff > 0                                                             # shape: (n_samples, n_samples)
        times_mask = torch.logical_and(times_mask, survival_events.unsqueeze(1) == 1)           # shape: (n_samples, n_samples)
        events_mask = survival_events.unsqueeze(0)                                              # shape: (1, n_samples)
        events_mask = events_mask.repeat(events_mask.shape[1], 1)                               # shape: (n_samples, n_samples)
        comp_mask = torch.logical_and(times_mask, events_mask)                                  # shape: (n_samples, n_samples)

        return comp_mask

    @staticmethod
    def comparable_times(predicted_times, comp_mask):
        """
        Computes all possible pairs of predicted survival times for the input samples and returns only the comparable ones.

        Parameters:
        - predicted_times: tensor with predicted survival times. Shape: (n_samples, 1).
        - comp_mask: tensor with a mask for the comparable pairs. Shape: (n_samples, n_samples).
                     comp_mask[i, j] = 1 if the pair(survival_times[i], survival_time[j]) is comparable, 0 otherwise.
                     survival_times is aligned with predicted_times and stores the true survival times, i.e., survival_times[i] is the true survival time for predicted_times[i].
        
        Returns:
        - comp_pairs: tensor with the comparable pairs of predicted survival times. Shape: (n_comparable_pairs, 2).
        """

        # compute all possible pairs of predicted times
        times_1 = predicted_times.repeat(1, predicted_times.shape[0])                           # shape: (n_samples, n_samples)
        times_2 = predicted_times.reshape((1, predicted_times.shape[0]))                        # shape: (1, n_samples)
        times_2 = times_2.repeat(times_2.shape[1], 1)                                           # shape: (n_samples, n_samples)
        all_pairs = torch.stack([times_1, times_2], dim=-1)                                     # shape: (n_samples, n_samples, 2)

        # filter only the comparable pairs
        comp_pairs = all_pairs[comp_mask]                                                       # shape: (n_comparable_pairs, 2)

        return comp_pairs

    @staticmethod
    def compute_loss(model, dataloader, loss_fn=torch.nn.MarginRankingLoss, margin=1, device=torch.device('cpu')):
        """
        Computes the MarginRankingLoss for the input predicted survival times, true survival times and survival events.

        Parameters:
        - model: SurvivalGNN model to be used.
        - dataloader: DataLoader object with the input data.
        - loss_fn: loss function to be used. Default is torch.nn.MarginRankingLoss.
        - margin: margin to be used in the loss function.
        - device: device to be used for tensor operations.

        Returns:
        - loss: MarginRankingLoss computed for the input data.
        """

        # set the model to evaluation mode
        model.eval()

        # define the loss function
        loss_function = loss_fn(margin=margin)

        # initialize the loss to 0
        loss = 0

        # iterate through batches
        for features, true_times, survival_events in dataloader:

            # move the batch to the device
            features = features.to(device)
            true_times = true_times.to(device)
            survival_events = survival_events.to(device)

            # forward pass to predicted times for all graphs in the batch
            predicted_times = model(features.x, features.edge_index, features.batch)

            # compute the mask for the comparable pairs
            comp_mask = TrainerSurvival.comparable_mask(true_times, survival_events)

            # compute the comparable pairs of predicted survival times
            comp_pairs = TrainerSurvival.comparable_times(predicted_times, comp_mask)

            # split the tensor with comparable predicted times into two tensors with the first and second elements of each pair
            pred_times_1 = comp_pairs[:, 0]
            pred_times_2 = comp_pairs[:, 1]

            # create a tensor with all 1's, indicating that the first element of each pair is greater than the second
            first_higher = torch.ones(pred_times_1.shape[0]).to(device)

            # compute the loss
            if comp_pairs.shape[0] > 0:
                loss += loss_function(pred_times_1, pred_times_2, first_higher).item()

        # average the loss over the number of batches
        loss /= len(dataloader)

        return loss

    @staticmethod
    def predict(model, dataloader, device=torch.device('cpu')):
        """
        Predicts the survival times for the samples in the input dataloader.

        Parameters:
        - model: SurvivalGNN model to be used.
        - dataloader: DataLoader object with the input data.
        - device: device to be used for tensor operations.

        Returns:
        - predicted_times: tensor with the predicted survival times for the samples in the input dataloader. Shape: (n_samples,).
        """
            
        # set the model to evaluation mode
        model.eval()

        # initialize the list to store the predicted times
        predicted_times = []

        # iterate through batches
        for features, true_times, survival_events in dataloader:

            # move the batch to the device
            features = features.to(device)

            # forward pass to predicted times for all graphs in the batch
            predicted_times_batch = model(features.x, features.edge_index, features.batch)

            # append the predicted times to the list
            predicted_times.append(predicted_times_batch)

        # concatenate the predicted times for all batches
        predicted_times = torch.cat(predicted_times, dim=0)

        return predicted_times

    @staticmethod
    def test_model(model, dataloader, device=torch.device('cpu')):
        """
        Computes the censored c-index for the input data.

        Parameters:
        - model: SurvivalGNN model to be used.
        - dataloader: DataLoader object with the input data.
        - device: device to be used for tensor operations.

        Returns:
        - censored_c_index: censored c-index computed for the input data.
        """

        # retrieve true survival times and survival events from the dataloader
        true_times = []
        survival_events = []
        for features, times, events in dataloader:
            true_times.append(times)
            survival_events.append(events)
        true_times = torch.cat(true_times, dim=0).squeeze().detach().cpu().numpy()
        survival_events = torch.cat(survival_events, dim=0).squeeze().detach().cpu().numpy()

        # predict survival times for the input data
        predicted_times = TrainerSurvival.predict(model, dataloader, device=device).squeeze().detach().cpu().numpy()

        # convert survival times into risk scores
        predicted_risks = -predicted_times

        # convert the survival events to boolean
        survival_events = survival_events.astype(np.bool)

        # compute the censored concordance index
        censored_c_index = concordance_index_censored(survival_events, true_times, predicted_risks)[0]
        
        return censored_c_index

    @staticmethod
    def train_epoch(model, train_dataloader, loss_fn=torch.nn.MarginRankingLoss, margin=1, opt=torch.optim.Adam, val_dataloader=None, device=torch.device('cpu')):
        """
        Trains a SurvivalGNN on the input training data for a single epoch.
        A validation set can be passed as input as well, in which case it is used for evaluation.

        Parameters:
        - model: SurvivalGNN model to be used.
        - train_dataloader: torch_geometric DataLoader with training data.
        - loss_fn: loss function to be used. Default is torch.nn.MarginRankingLoss.
        - margin: margin to be used in the loss function.
        - opt: optimizer to be used to train the tumorTreeGNN.
        - val_dataloader: torch_geometric DataLoader with validation data. If None, no validation is performed.
        - device: device to be used for tensor operations.

        Returns:
        - avg_epoch_loss: loss computed for training data during the current epoch, averaged across batches.
        - val_avg_loss: average loss computed for validation data.
                        None is returned if validation data is provided as input.
        """

        # set the model to train mode
        model.train()
        
        # set loss for the current epoch to 0
        epoch_loss = 0

        # define the loss function
        loss_function = loss_fn(margin=margin)
        
        # iterate through batches
        for features, survival_times, survival_events in train_dataloader:

            # move the batch to the device
            features = features.to(device)
            survival_times = survival_times.to(device)
            survival_events = survival_events.to(device)
            
            # reset gradients computed for the previous batch to 0
            opt.zero_grad()

            # forward pass to predicted times for all graphs in the batch
            predicted_times = model(features.x, features.edge_index, features.batch)

            # extract only comparable pairs from all possible pairs of times in predicted_times, where a pair (predicted_times[i], predicted_times[j]) is comparable if
            # survival_times[i] > survival_times[j] and survival_events[j] == 1
            compable_mask = TrainerSurvival.comparable_mask(survival_times, survival_events)
            compable_predicted_times = TrainerSurvival.comparable_times(predicted_times, compable_mask)

            # split the tensor with comparable predicted times into two tensors with the first and second elements of each pair
            pred_times_1 = compable_predicted_times[:, 0]
            pred_times_2 = compable_predicted_times[:, 1]

            # create a tensor with all 1's, indicating that the first element of each pair is greater than the second
            first_higher = torch.ones(pred_times_1.shape[0]).to(device)

            # check that the batch contains at least one comparable pair
            if compable_predicted_times.shape[0] > 0:

                # compute the loss
                loss = loss_function(pred_times_1, pred_times_2, first_higher)
                
                # backpropagate
                loss.backward()

                # update the weights
                opt.step()

                # add the loss for the current batch to the current value of the loss for the current epoch
                epoch_loss += loss.item()

        # average the epoch loss over the number of batches
        avg_epoch_loss = epoch_loss / len(train_dataloader)

        # evaluate the model on the validation set, if needed
        val_avg_loss = None
        if val_dataloader is not None:
            val_avg_loss = TrainerSurvival.compute_loss(model, val_dataloader, loss_fn=loss_fn, margin=margin, device=device)

        return avg_epoch_loss, val_avg_loss

    @staticmethod
    def train(
            model,
            train_data,
            optimizer=torch.optim.Adam,
            weight_decay=0,
            loss_fn=torch.nn.MarginRankingLoss,
            margin=1,
            batch_size=16,
            val_data=None,
            plot_save=None,
            verbose=True,
            epochs=100,
            lr=0.001,
            early_stopping_tolerance=None,
            device=torch.device('cpu'),
            save_model=None
            ):
        """
        Trains a SurvivalGNN on the input training data, printing and plotting information, if required.
        A validation set can be passed as input as well, in which case it is included in the printed and plotted information.

        Parameters:
        - model: SurvivalGNN model to be used.
        - train_data: TorchSurvivalDataset object with training data.
        - optimizer: optimizer to be used to train the SurvivalGNN.
        - weight_decay: weight decay to be used in the optimizer.
        - loss_fn: loss function to be used. Default is torch.nn.MarginRankingLoss.
        - margin: margin to be used in the loss function.
        - batch_size: batch size.
        - val_data: TorchSurvivalDataset object with validation data. If None, no validation is performed.
        - plot_save: path where to save the plot with losses. If None, then no plot is created.
        - verbose: True if training information must be print, False otherwise.
        - epochs: number of training iterations.
        - lr: learning rate.
        - early_stopping_tolerance: number of epochs without improvement after which the training process is stopped. If None, early stopping is not applied.
        - device: device to be used for tensor operations.
        - save_model: path where to save the trained model. If None, the model is not saved.
        """

        # flag indicating whether a validation set is provided as input
        val_flag = True
        if val_data is None:
            val_flag = False
        
        # create dataloaders
        train_dl = TrainerSurvival.get_dataloader(train_data, batch_size=batch_size)
        val_dl = None
        if val_flag:
            val_dl = TrainerSurvival.get_dataloader(val_data, batch_size=batch_size, shuffle=False)
            
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
            train_loss, val_loss = TrainerSurvival.train_epoch(model, train_dl, loss_fn=loss_fn, margin=margin, opt=opt, val_dataloader=val_dl, device=device)

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
                TrainerSurvival.plot_losses([train_losses_epochs_list, val_losses_epochs_list], plot_save, ['Train', 'Validation'], 'Losses over Epochs')
            else:
                TrainerSurvival.plot_losses([train_losses_epochs_list], plot_save, ['Train'], 'Loss over Epochs')
        
        # save the trained model, if required
        if save_model is not None:
            os.makedirs(os.path.dirname(save_model), exist_ok=True)
            torch.save(model.state_dict(), save_model)
    
    @staticmethod
    def train_val_split(phylogenies, survival_df, val_proportion=0.2, rd_seed=None):
        """
        Randomly splits the input phylogenies and survival data into training and validation sets.

        Parameters:
        - phylogenies: dictionary with phylogenies for each patient. Keys are patient IDs.
        - survival_df: pandas DataFrame with survival data for each patient. Must contain the column 'Patient_ID'.
        - val_proportion: float representing the proportion of patients to be included in the validation set.
        - rd_seed: integer with the random seed for reproducibility. Used for all random operations.

        Returns:
        - train_phylogenies: dictionary with phylogenies for each patient in the training set. Keys are patient IDs.
        - val_phylogenies: dictionary with phylogenies for each patient in the validation set. Keys are patient IDs.
        - train_survival: pandas DataFrame with survival data for each patient in the training set.
        - val_survival: pandas DataFrame with survival data for each patient in the validation set.
        """

        # extract and shuffle the patient IDs
        patients_ids = list(phylogenies.keys())
        if rd_seed is not None:
            np.random.seed(rd_seed)
        np.random.shuffle(patients_ids)

        # split the patient IDs into training and validation sets
        n_val = int(val_proportion * len(patients_ids))
        val_ids = patients_ids[:n_val]
        train_ids = patients_ids[n_val:]

        # split the phylogenies into training and validation sets
        train_phylogenies = {key: phylogenies[key] for key in train_ids}
        val_phylogenies = {key: phylogenies[key] for key in val_ids}

        # split the survival data into training and validation sets
        train_survival = survival_df[survival_df['Patient_ID'].isin(train_ids)]
        val_survival = survival_df[survival_df['Patient_ID'].isin(val_ids)]

        return train_phylogenies, val_phylogenies, train_survival, val_survival
    
    @staticmethod
    def load_train_data(phylogenies_path, survival_path, survival_time_label, survival_event_label, rd_seed=None, min_label_occurrences=0, node_encoding_type='clone'):
        """
        Loads the training datasets with phylogenies and survival data and merges them into a TorchSurvivalDataset.
        It also returns the number of labels present in the training set, necessary to define the input size of the SurvivalGNN.

        Parameters:
        - phylogenies_path: string with the path to the file with phylogenies.
        - survival_path: string with the path to the file with survival data.
        - survival_time_label: string with the label for the survival time in the survival data.
        - survival_event_label: string with the label for the survival event in the survival data.
        - rd_seed: integer with the random seed for reproducibility. Used for all random operations.
        - survival_time_label: string with the label for the survival time in the survival data.
        - survival_event_label: string with the label for the survival event in the survival data.
        - min_label_occurrences: integer with the minimum number of occurrences for a label to be considered in the training set.
        - node_encoding_type: string with the type of encoding to be used for the nodes in the phylogenies. Can be either 'mutation' or 'clone'.

        Returns:
        - train_dataloader: TorchSurvivalDataset object with the training data.
        - n_labels: number of labels present in the training set.
        """

        # load the dataset with phylogenies
        train_phylogenies = TrainerTumorModel.load_dataset_txt(phylogenies_path)

        # load the dataset with survival data
        train_survival = pd.read_csv(survival_path)

        # create a TumorDataset object for the training set that contains patients sorted by patient id so to allow for reproducibility
        train_sorted_keys = sorted(train_phylogenies.keys())
        train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
        train_data = TumorDataset(train_list_patients)

        # create a train array of tuples with survival time and survival event for each patient using the sorted keys
        sorted_train_df = train_survival.sort_values(by='Patient_ID')
        train_survival = np.array([sorted_train_df[survival_time_label].values, sorted_train_df[survival_event_label].values])

        # compute the set of labels to be considered, based on the number of occurrences in the training set
        if min_label_occurrences > 0:
            train_data.remove_infreq_labels(min_label_occurrences)

        # sample one graph per patient
        train_data.sample_one_graph_per_patient(rd_seed=rd_seed)

        # create the TorchSurvivalDataset for the training set
        train_torch_data = TorchSurvivalDataset(train_data, train_survival, node_encoding_type=node_encoding_type)

        # number of labels that appear in the training set
        n_labels = len(train_data.node_labels())

        return train_torch_data, n_labels

    @staticmethod
    def load_train_val_data(phylogenies_path, survival_path, survival_time_label, survival_event_label, val_proportion=0.2, rd_seed=None, min_label_occurrences=0, node_encoding_type='clone', batch_size=64):
        """
        Loads the training and validation datasets with phylogenies and survival data, and creates the corresponding dataloaders.
        It also returns the number of labels present in the training set, necessary to define the input size of the SurvivalGNN.

        Parameters:
        - phylogenies_path: string with the path to the file with phylogenies.
        - survival_path: string with the path to the file with survival data.
        - survival_time_label: string with the label for the survival time in the survival data.
        - survival_event_label: string with the label for the survival event in the survival data.
        - val_proportion: float with the proportion of patients to be inserted in the validation set.
        - rd_seed: integer with the random seed for reproducibility. Used for all random operations.
        - min_label_occurrences: integer with the minimum number of occurrences for a label to be considered in the training set.
        - node_encoding_type: string with the type of encoding to be used for the nodes in the phylogenies. Can be either 'mutation' or 'clone'.
        - batch_size: batch size to be used for the dataloaders.

        Returns:
        - train_dataloader: DataLoader object with the training data.
        - val_dataloader: DataLoader object with the validation data.
        - n_labels: number of labels present in the training set.
        """

        # load the dataset with phylogenies
        phylogenies = TrainerTumorModel.load_dataset_txt(phylogenies_path)

        # load the dataset with survival data
        survival = pd.read_csv(survival_path)

        # split the datasets into training and validation sets
        train_phylogenies, val_phylogenies, train_survival, val_survival = TrainerSurvival.train_val_split(phylogenies, survival, val_proportion, rd_seed)

        # create TumorDataset objects for training and validation sets that contains patients sorted by patient id so to allow for reproducibility
        train_sorted_keys = sorted(train_phylogenies.keys())
        train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
        train_data = TumorDataset(train_list_patients)

        val_sorted_keys = sorted(val_phylogenies.keys())
        val_list_patients = [val_phylogenies[key] for key in val_sorted_keys]
        val_data = TumorDataset(val_list_patients)

        # create train and validation arrays of tuples with survival time and survival event for each patient using the sorted keys
        sorted_train_df = train_survival.sort_values(by='Patient_ID')
        train_survival = np.array([sorted_train_df[survival_time_label].values, sorted_train_df[survival_event_label].values])
        sorted_val_df = val_survival.sort_values(by='Patient_ID')
        val_survival = np.array([sorted_val_df[survival_time_label].values, sorted_val_df[survival_event_label].values])

        # compute the set of labels to be considered, based on the number of occurrences in the training set
        if min_label_occurrences > 0:
            train_data.remove_infreq_labels(min_label_occurrences)
            val_data.replace_label_set(train_data.node_labels(), replace_with='empty')

        # sample one graph per patient
        train_data.sample_one_graph_per_patient(rd_seed=rd_seed)
        val_data.sample_one_graph_per_patient(rd_seed=rd_seed)

        # create the TorchSurvivalDataset objects for training and val sets
        train_torch_data = TorchSurvivalDataset(train_data, train_survival, node_encoding_type=node_encoding_type)
        val_torch_data = TorchSurvivalDataset(val_data, val_survival, node_encoding_type=node_encoding_type, known_labels_mapping=train_torch_data.node_labels_mapping)

        # create the dataloaders
        train_dataloader = TrainerSurvival.get_dataloader(train_torch_data, batch_size=batch_size)
        val_dataloader = TrainerSurvival.get_dataloader(val_torch_data, batch_size=batch_size, shuffle=False)

        # number of labels that appear in the training set
        n_labels = len(train_data.node_labels())

        return train_dataloader, val_dataloader, n_labels

    @staticmethod
    def tuning_objective(trial, phylogenies_path, survival_path, validation_proportion, survival_time_label, survival_event_label, random_seed=None, device=torch.device('cpu')):
        """
        Objective function to be used in the hyperparameter tuning process.

        Parameters:
        - trial: optuna.Trial object to be used for hyperparameter tuning.
        - phylogenies_path: string with the path to the file with phylogenies.
        - survival_path: string with the path to the file with survival data.
        - validation_proportion: float with the proportion of patients to be inserted in the validation set.
        - survival_time_label: string with the label for the survival time in the survival data.
        - survival_event_label: string with the label for the survival event in the survival data.
        - random_seed: integer with the random seed for reproducibility. Used for all random operations.
        - device: device to be used for tensor operations.

        Returns:
        - censored_c_index: censored c-index computed by the trained model on the validation set.
        """

        # suggest hyperparameters
        weight_decay = trial.suggest_float('weight_decay', 1e-06, 1e-01, log=True)
        margin = trial.suggest_categorical('margin', [1.0])
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        loss_fn = trial.suggest_categorical('loss_fn', ['SquaredMarginRankingLoss'])
        optimizer = trial.suggest_categorical('optimizer', ['Adam'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        epochs = trial.suggest_categorical('epochs', [30, 50, 100, 200, 300])
        batch_normalization = trial.suggest_categorical('batch_normalization', [True, False])
        dropout_prob_1 = trial.suggest_float('dropout_prob_1', 0.0, 0.8, step=0.1)
        dropout_prob_2 = trial.suggest_float('dropout_prob_2', 0.0, 0.8, step=0.1)
        dropout_prob_3 = trial.suggest_float('dropout_prob_3', 0.0, 0.8, step=0.1)
        h_1 = trial.suggest_categorical('h_1', [16, 32, 64, 128])
        h_2 = trial.suggest_categorical('h_2', [16, 32, 64, 128])
        hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        min_label_occurrences = trial.suggest_categorical('min_label_occurrences', [0])
        node_encoding_type = trial.suggest_categorical('node_encoding_type', ['clone'])

        # load the training and validation data and get the set of labels in the training set
        train_dataloader, val_dataloader, n_labels = TrainerSurvival.load_train_val_data(
            phylogenies_path,
            survival_path,
            survival_time_label,
            survival_event_label,
            validation_proportion,
            random_seed,
            min_label_occurrences,
            node_encoding_type,
            batch_size
        )

        # create a SurvivalGNN instance with input size based on the labels in the dataset
        model = SurvivalGNN(
            n_node_labels=n_labels,
            h_1_dim=h_1,
            h_2_dim=h_2,
            hidden_dim=hidden_dim,
            dropout_prob_1=dropout_prob_1,
            dropout_prob_2=dropout_prob_2,
            dropout_prob_3=dropout_prob_3,
            batch_normalization=batch_normalization,
            device=device
        )

        # initialize the optimizer
        optimizer = Utils.select_optimizer(optimizer)
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

        # initialize the loss function
        loss_fn = TrainerSurvival.select_loss_survival(loss_fn)
    
        # iterate through epochs
        for epoch in range(epochs):

            # train the model and compute losses
            train_loss, val_loss = TrainerSurvival.train_epoch(model, train_dataloader, loss_fn=loss_fn, margin=margin, opt=opt, val_dataloader=val_dataloader, device=device)

            # compute the c-index for the validation set
            censored_c_index = TrainerSurvival.test_model(model, val_dataloader, device=device)

            # report the computed c-index to optuna
            trial.report(censored_c_index, epoch)

            # check if the training process needs to be stopped because the trial is not promising
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        # compute the c-index for the validation set
        censored_c_index = TrainerSurvival.test_model(model, val_dataloader, device=device)

        return censored_c_index

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

class Survival_Analysis:
    """
    Class with static functions to apply survival analysis.
    """

    @staticmethod
    def kaplan_meier_curves(data, time_column, event_column, save_path):
        """
        Creates and saves a facet figure with a Kaplan-Meier plot for each clustering size and a curve for each cluster in each plot.
        It also plots the p-values for the log-rank test comparing the hazard functions referred to different clusters in each clustering.

        Parameters:
        - data: dataframe with columns: 'Patient_ID', time_column, event_column, 'K_2', 'K_3', ..., 'K_n', where K_i is the cluster label assigned
                to the patient for the clustering with size i.
        - time_column: string with the name of the column with the time of the event.
        - event_column: string with the name of the column with the event indicator.
        - save_path: string with the path where to save the plot.
        """

        # map event 1 to True and event 0 to False
        data[event_column] = data[event_column].map({1: True, 0: False})

        # use the Kaplan-Meier estimator to estimate the survival function for each cluster in each clustering
        kaplan_meier_data = []
        for clustering in data.filter(like='K_').columns:
            for cluster in data[clustering].unique():
                curr_samples = data[data[clustering] == cluster]
                time, survival_prob, conf_int = kaplan_meier_estimator(curr_samples[event_column], curr_samples[time_column], conf_type="log-log")
                for i in range(len(time)):
                    kaplan_meier_data.append({
                        'K': int(clustering[2:]),
                        'Cluster': cluster,
                        'Time': time[i],
                        'Survival_Prob': survival_prob[i],
                        'Conf_Int_Min': conf_int[0][i],
                        'Conf_Int_Max': conf_int[1][i]
                    })
        kaplan_meier_data = pd.DataFrame(kaplan_meier_data)
        
        # perform the log-rank test for each clustering
        structured_data = Surv.from_dataframe(event_column, time_column, data)
        log_rank_tests = []
        for clustering in data.filter(like='K_').columns:
            curr_test = compare_survival(structured_data, data[clustering]) 
            log_rank_tests.append({
                'K': int(clustering[2:]),
                'P_Value': curr_test[1]
            })
        log_rank_tests = pd.DataFrame(log_rank_tests)

        # set theme, style and add a light grid
        sns.set_theme()
        sns.set_style('white')
        

        # set color palette
        palette = sns.color_palette('tab10', n_colors=len(kaplan_meier_data['Cluster'].unique()))

        # create a facet figure with a facet for each clustering size
        grid = sns.FacetGrid(kaplan_meier_data, col='K', hue='Cluster', col_wrap=3, palette=palette, height=4, aspect=1)

        # plot Kaplan-Meier confidence intervals
        grid.map(plt.fill_between, 'Time', 'Conf_Int_Min', 'Conf_Int_Max', alpha=0.25, step='post', zorder=1)

        # plot Kaplan-Meier curves
        grid.map(plt.step, 'Time', 'Survival_Prob', where='post', zorder=4)

        # set axes, titles and legend
        grid.set_axis_labels('Time (months)', 'Survival Probability')
        grid.set(xlim=(0, None), ylim=(0, 1))
        grid.set_titles(col_template='K={col_name}')
        grid.add_legend(title='Cluster')

        # iterate through each facet so to add log-rank p-values and median survival annotations
        for ax, k in zip(grid.axes.flat, sorted(log_rank_tests['K'].unique())):
            
            # add annotations with log-rank p-values
            p_val = log_rank_tests[log_rank_tests['K'] == k]['P_Value'].values[0]
            ax.text(0.49, 0.87, f'Log-rank p = {p_val:.1e}', transform=ax.transAxes)

            # add 0.5 to y-ticks if not present
            yticks = ax.get_yticks()
            if 0.5 not in yticks:
                yticks = list(yticks) + [0.5]
            ax.set_yticks(yticks)

            # set only the 0.5 y-tick label to be bold
            for tick_label in ax.get_yticklabels():
                if tick_label.get_text() == '0.5':
                    tick_label.set_fontweight('bold')
            
            # add a custom grid without spines
            for y_tick in yticks:
                if y_tick != 0.5:
                    ax.plot([0, 400], [y_tick, y_tick], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
            xticks = ax.get_xticks()
            for x_tick in xticks:
                ax.plot([x_tick, x_tick], [0, 1], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

            # compute the maximum median survival time
            max_median_time = 0
            for cluster in sorted(kaplan_meier_data[kaplan_meier_data['K'] == k]['Cluster'].unique()):
                cluster_data = kaplan_meier_data[(kaplan_meier_data['K'] == k) & (kaplan_meier_data['Cluster'] == cluster)]
                median_time = cluster_data[cluster_data['Survival_Prob'] <= 0.5]['Time'].min()
                if median_time > max_median_time:
                    max_median_time = median_time

            # add a horiontal line at 0.5 survival probability that stops at the maximum median survival time
            ax.plot([0, max_median_time], [0.5, 0.5], linestyle='dashed', color='black', alpha=0.5, zorder=2)

            # for each cluster, add a vertical line intersecting the horizontal line at the median survival time and highlight the point
            for i, cluster in enumerate(sorted(kaplan_meier_data[kaplan_meier_data['K'] == k]['Cluster'].unique())):
                cluster_data = kaplan_meier_data[(kaplan_meier_data['K'] == k) & (kaplan_meier_data['Cluster'] == cluster)]
                median_time = cluster_data[cluster_data['Survival_Prob'] <= 0.5]['Time'].min()
                ax.plot([median_time, median_time], [0.0, 0.5], linestyle='dashed', color=palette[i], zorder=3)
                ax.scatter(median_time, 0.5, color=palette[i], edgecolor='black', alpha=0.8, zorder=5)

        # add a supertitle and tight layout
        plt.suptitle('Kaplan-Meier Curves for Different Clustering Sizes')
        grid.tight_layout()

        # save the figure
        plt.savefig(save_path)

        # close the figure
        plt.close()
    
    @staticmethod
    def kaplan_meier_plot_for_k(data, time_column, event_column, k, save_path):
        """
        Creates and saves a Kaplan-Meier plot for a specific clustering size `k`.
        It also plots the p-value for the log-rank test comparing the hazard functions for different clusters in that clustering.

        Parameters:
        - data: DataFrame with columns: 'Patient_ID', time_column, event_column, 'K_2', 'K_3', ..., 'K_n'.
        - time_column: string, name of the column with time-to-event.
        - event_column: string, name of the event indicator column.
        - k: int, clustering size to plot (e.g., 2, 3, 4, etc.).
        - save_path: string, path to save the plot.
        """

        # map event 1 to True and 0 to False
        data[event_column] = data[event_column].map({1: True, 0: False})

        clustering_col = f'K_{k}'

        # compute Kaplan-Meier survival curves for each cluster
        km_data = []
        for cluster in sorted(data[clustering_col].unique()):
            subset = data[data[clustering_col] == cluster]
            time, survival_prob, conf_int = kaplan_meier_estimator(subset[event_column], subset[time_column], conf_type="log-log")
            for i in range(len(time)):
                km_data.append({
                    'Cluster': cluster,
                    'Time': time[i],
                    'Survival_Prob': survival_prob[i],
                    'Conf_Int_Min': conf_int[0][i],
                    'Conf_Int_Max': conf_int[1][i]
                })
        km_data = pd.DataFrame(km_data)

        # log-rank test
        structured_data = Surv.from_dataframe(event_column, time_column, data)
        p_value = compare_survival(structured_data, data[clustering_col])[1]

        # plotting
        sns.set_theme()
        sns.set_style("whitegrid")
        palette = sns.color_palette('tab10', n_colors=len(km_data['Cluster'].unique()))
        fig, ax = plt.subplots(figsize=(6, 5))

        for i, cluster in enumerate(sorted(km_data['Cluster'].unique())):
            cluster_data = km_data[km_data['Cluster'] == cluster]
            ax.fill_between(cluster_data['Time'], cluster_data['Conf_Int_Min'], cluster_data['Conf_Int_Max'],
                            step='post', alpha=0.25, color=palette[i])
            ax.step(cluster_data['Time'], cluster_data['Survival_Prob'], where='post',
                    label=f'Cluster {cluster}', color=palette[i])

            # median survival
            median_time = cluster_data[cluster_data['Survival_Prob'] <= 0.5]['Time'].min()
            if pd.notna(median_time):
                ax.axvline(median_time, ymin=0, ymax=0.5, linestyle='--', color=palette[i], alpha=0.8)
                ax.scatter(median_time, 0.5, color=palette[i], edgecolor='black', zorder=5)

        # horizontal line at 0.5
        max_median = km_data[km_data['Survival_Prob'] <= 0.5].groupby('Cluster')['Time'].min().max()
        ax.axhline(0.5, xmax=max_median / ax.get_xlim()[1], linestyle='--', color='black', alpha=0.5)

        ax.set_title(f'Kaplan-Meier Curve (K={k})')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(0, 1)
        ax.legend(title='Cluster')
        ax.text(0.5, 0.9, f'Log-rank p = {p_value:.1e}', transform=ax.transAxes, ha='center')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def pairwise_logranks(data, time_column, event_column):
        """
        Computes the pairwise log-rank p-values for each clustering size.

        Parameters:
        - data: dataframe with columns: 'Patient_ID', time_column, event_column, 'K_2', 'K_3', ..., 'K_n', where K_i is the cluster label assigned
                to the patient for the clustering with size i.
        - time_column: string with the name of the column with the time of the event.
        - event_column: string with the name of the column with the event indicator.

        Returns:
        - p_values: dictionary with the clustering sizes as keys and a dictionary with each pairwise log-rank p-values as values.
        """
        
        # perform the log-rank test for each pair of clusters in each clustering
        p_values = {}
        for clustering in [c for c in data.columns if c.startswith('K_')]:
            p_values[clustering] = {}
            clusters = data[clustering].unique()
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    curr_data = data[(data[clustering] == clusters[i]) | (data[clustering] == clusters[j])]
                    structured_data = Surv.from_dataframe(event_column, time_column, curr_data)
                    curr_test = compare_survival(structured_data, curr_data[clustering])
                    p_values[clustering][f'{clusters[i]} vs {clusters[j]}'] = curr_test[1]
        
        return p_values
    
    @staticmethod
    def print_logranks(p_values):
        """
        Prints the pairwise log-rank p-values for each clustering size.

        Parameters:
        - p_values: dictionary with the clustering sizes as keys and a dictionary with each pairwise log-rank p-values as values.
        """

        # iterate through all items in the dictionary and print the p-values
        for clustering, tests in p_values.items():
            print(f'\nClustering with {clustering[2:]} clusters:')
            for pair, p_value in tests.items():
                print(f'{pair}: {p_value:.1e}')

class Survival_Prediction:
    """
    Class with static functions for predicting survival time and dealing with predictions.
    """

    @staticmethod
    def train_predictor(predictor, features, targets, hyperparam_grid, cv_folds, n_jobs=1):
        """
        Trains the input predictor on the input features to predict the input targets.
        It performs a grid search to tune the hyperparameters of the predictor, using the input hyperparameter_grid.

        Parameters:
        - predictor: predictor object with a fit method.
        - features: numpy array with the input features.
        - targets: structured array with the true survival times and event indicators.
        - hyperparam_grid: dictionary with the hyperparameters to tune and their possible values.
        - cv_folds: integer with the number of cross-validation folds to use.
        - n_jobs: integer with the number of jobs to run in parallel.

        Returns:
        - best_predictor: predictor object with the best hyperparameters found during the grid search.
        """

        # create a GridSearchCV instance
        grid_search = GridSearchCV(
            estimator=predictor,
            param_grid=hyperparam_grid,
            cv=cv_folds,
            verbose=3,
            n_jobs=n_jobs
        )

        # perform grid search to tune the hyperparameters and train the predictor on the input features
        grid_search.fit(features, targets)

        # extract the predictor with the highest score
        best_predictor = grid_search.best_estimator_

        # print the best hyperparameters found
        print(f'Best hyperparameters: {grid_search.best_params_}')

        return best_predictor

    @staticmethod
    def evaluate_ssvm(predictor_name, predictor, features, train_targets, test_targets, experiment_id=27, regression=False):
        """
        Evaluates the input survival SVM on the input features with true values in test_targets.
        In particular, it computes the concordance index and the concordance index IPCW.
        It is not possible to compute the integrated Brier score for survival SVM models.

        Parameters:
        - predictor_name: string with the name of the survival SVM predictor.
        - predictor: predictor object.
        - features: numpy array with the input features.
        - train_targets: structured array with the true survival times and event indicators used to train the predictor. It is used to estimate the censoring distribution.
        - test_targets: structured array with the true survival times and event indicators to be compared with predictions.
        - experiment_id: integer with the id of the experiment.
        - regression: boolean indicating whether the survival SVM is trained considering also the regression objective or not.

        Returns:
        - evaluation_df: pandas dataframe with the evaluation metrics computed for the predictor on the input features.
        """

        # compute the c-index on the test set
        score_cindex = predictor.score(features, test_targets)
        
        # ssvms trained considering also the regression objective predict a ranking of the patients based on survival times, but the c-index must be computed on risks
        if regression:
            
            # predict the survival times on the test set
            predicted_time_ranks = predictor.predict(features)      # the lower the rank, the lower the survival time and so the larger the risk
            # invert the ranks
            predicted_risk_ranks = -predicted_time_ranks            # the lower the rank, the larger the survival time and so the lower the risk

        # ssvms trained only on the ranking objective predict the ranks based on risk scores directly
        else:
            
            # predict the ranks of risk scores on the test set
            predicted_risk_ranks = predictor.predict(features)
        
        # compute the c-index with IPCW on the test set
        score_cindex_ipcw = concordance_index_ipcw(train_targets, test_targets, predicted_risk_ranks)[0]

        evaluation_df = pd.DataFrame(
            [{
                'Experiment ID': experiment_id,
                'Predictor': predictor_name,
                'C-Index Censored': score_cindex,
                'C-Index IPCW': score_cindex_ipcw
            }]
        )

        return evaluation_df

    @staticmethod
    def c_index_plot(evaluation_df, save_path):
        """
        Creates and saves a box plot with the c-index scores for each predictor in the input evaluation dataframe.

        Parameters:
        - evaluation_df: pandas dataframe with the evaluation metrics computed for the predictor on the input features.
                         It must contain the columns: 'Predictor', 'C-Index Censored', 'C-Index IPCW', 'Integrated Brier Score', 'Experiment ID'.
        - save_path: string with the path where to save the plot.
        """

        # remove the 'Random' predictor and the 'Kaplan-Meier' predictor from the dataframe
        plot_df = evaluation_df[evaluation_df['Predictor'] != 'Random']
        plot_df = plot_df[plot_df['Predictor'] != 'Kaplan-Meier']

        # keep just the columns 'Experiment ID', 'Predictor' and 'C-Index Censored'
        plot_df = plot_df[['Experiment ID', 'Predictor', 'C-Index Censored']]

        # rename the predictor names
        plot_df['Predictor'] = plot_df['Predictor'].replace({'Random Survival Forest': 'Tumor Graph RSF'})

        # set theme, style and add a light grid
        sns.set_theme()
        sns.set_style('whitegrid')

        # set the font size
        sns.set_context('paper', font_scale=1.5)

        # rename the predictor names so to have them displayed better in the plot
        plot_df['Predictor'] = plot_df['Predictor'].replace(
            {
                'Supervised GNN': 'Supervised\nGNN',
                'GNN SSVM': 'Unsupervised\nGNN',
                'Baseline SSVM': 'Baseline'
            }
        )

        # create a box plot with the c-index scores for each predictor
        ax = sns.boxplot(data=plot_df, x='Predictor', y='C-Index Censored', hue='Predictor', palette='tab10', linewidth=1.5)
        ax.set_ylim(0.4, 0.75)
        ax.set_xlabel('Predictor', labelpad=10)
        ax.set_ylabel('C-Index Censored', labelpad=10)

        # add a title and tight layout
        # plt.title('C-Index Censored for Different Predictors')
        plt.tight_layout()

        # save the plot
        plt.savefig(save_path)

        # close the figure
        plt.close()
    
    @staticmethod
    def c_index_ipcw_plot(evaluation_df, save_path):
        """
        Creates and saves a box plot with the c-index IPCW scores for each predictor in the input evaluation dataframe.

        Parameters:
        - evaluation_df: pandas dataframe with the evaluation metrics computed for the predictor on the input features.
                         It must contain the columns: 'Predictor', 'C-Index Censored', 'C-Index IPCW', 'Integrated Brier Score', 'Experiment ID'.
        - save_path: string with the path where to save the plot.
        """

        # remove the 'Random' predictor and the 'Kaplan-Meier' predictor from the dataframe
        plot_df = evaluation_df[evaluation_df['Predictor'] != 'Random']
        plot_df = plot_df[plot_df['Predictor'] != 'Kaplan-Meier']

        # keep just the columns 'Experiment ID', 'Predictor' and 'C-Index IPCW'
        plot_df = plot_df[['Experiment ID', 'Predictor', 'C-Index IPCW']]

        # rename the predictor names
        plot_df['Predictor'] = plot_df['Predictor'].replace({'Random Survival Forest': 'Tumor Graph RSF'})

        # set theme, style and add a light grid
        sns.set_theme()
        sns.set_style('whitegrid')

        # set the font size
        sns.set_context('paper', font_scale=1.5)

        # rename the predictor names so to have them displayed better in the plot
        plot_df['Predictor'] = plot_df['Predictor'].replace(
            {
                'Supervised GNN': 'Supervised\nGNN',
                'GNN SSVM': 'Unsupervised\nGNN',
                'Baseline SSVM': 'Baseline'
            }
        )

        # create a box plot with the c-index scores for each predictor
        ax = sns.boxplot(data=plot_df, x='Predictor', y='C-Index IPCW', hue='Predictor', palette='tab10')
        ax.set_ylim(0.4, 0.75)
        ax.set_xlabel('Predictor', labelpad=10)
        ax.set_ylabel('C-Index IPCW', labelpad=10)

        # add a title and tight layout
        # plt.title('C-Index IPCW for Different Predictors')
        plt.tight_layout()

        # save the plot
        plt.savefig(save_path)

        # close the figure
        plt.close()

    @staticmethod
    def c_index_facet_plot(evaluation_df, save_path):
        """
        Creates and saves a faceted plot with two subplots: one with the censored c-index and the other with the IPCW c-index.

        Parameters:
        - evaluation_df: pandas dataframe with the evaluation metrics computed for the predictor on the input features.
                         It must contain the columns: 'Predictor', 'C-Index Censored', 'C-Index IPCW', 'Experiment ID'.
        - save_path: string with the path where to save the plot.
        """

        # make the datagrame tidy
        plot_df = evaluation_df.melt(id_vars=['Predictor', 'Experiment ID'], value_vars=['C-Index Censored', 'C-Index IPCW'], var_name='Metric', value_name='Score')

        # change the names of the predictors
        plot_df = plot_df.replace(
            {
                'Supervised GNN': 'Supervised\nGNN',
                'GNN SSVM': 'Unsupervised\nGNN',
                'Baseline SSVM': 'Baseline'
            }
        )

        # set theme and style
        sns.set_theme('paper')
        sns.set_style('whitegrid')

        # set font sizes
        label_fontsize = 9
        title_fontsize = 10
        # suptitle_fontsize = 12

        # initialize a grid of plots with axes for each metric
        grid = sns.FacetGrid(plot_df, col="Metric", hue="Predictor", palette="tab10", col_wrap=2, height=2.5, aspect=1.2)

        # add a boxplot for each predictor and metric
        grid.map(sns.boxplot, "Predictor", "Score", order=plot_df['Predictor'].unique(), hue_order=plot_df['Predictor'].unique())

        # add a title to each subplot
        grid.set_titles(col_template="{col_name}", size=title_fontsize)

        # set the sizes of the axes labels
        for ax in grid.axes.flat:
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

        # set the limit for the y-axis
        for ax in grid.axes.flat:
            ax.set_ylim(0.38, 0.72)

        # tight layout
        # plt.suptitle('C-Index Scores for Different Predictors', fontsize=suptitle_fontsize)
        plt.tight_layout()

        # save the plot
        plt.savefig(save_path)

        # close the figure
        plt.close()

    @staticmethod
    def brier_score_plot(evaluation_df, save_path):
        """
        Creates and saves a box plot with the integrated brier scores for each predictor in the input evaluation dataframe.

        Parameters:
        - evaluation_df: pandas dataframe with the evaluation metrics computed for the predictor on the input features.
                         It must contain the columns: 'Predictor', 'C-Index Censored', 'C-Index IPCW', 'Integrated Brier Score', 'Experiment ID'.
        - save_path: string with the path where to save the plot.
        """

        # remove the 'Random' predictor from the dataframe
        plot_df = evaluation_df[evaluation_df['Predictor'] != 'Random']

        # keep just the columns 'Experiment ID', 'Predictor' and 'Integrated Brier Score'
        plot_df = plot_df[['Experiment ID', 'Predictor', 'Integrated Brier Score']]

        # rename the predictors' name
        plot_df['Predictor'] = plot_df['Predictor'].replace({'Random Survival Forest': 'Tumor Graph RSF'})

        # set theme, style and add a light grid
        sns.set_theme()
        sns.set_style('whitegrid')

        # create a box plot with the c-index scores for each predictor
        sns.boxplot(data=plot_df, x='Predictor', y='Integrated Brier Score', hue='Predictor', palette='tab10')

        # add a title and tight layout
        plt.title('Integrated Brier Score for Different Predictors')
        plt.tight_layout()

        # save the plot
        plt.savefig(save_path, dpi=300)

        # close the figure
        plt.close()

class Survival_Features:
    """
    Class with static functions to compute features for survival prediction.
    """

    @staticmethod
    def get_embeddings(
        model,
        train_torch_data,
        test_torch_data,
        train_sorted_keys,
        test_sorted_keys,
        batch_size=16,
        device=torch.device('cpu')
        ):
        """
        Computes the embeddings for the input training and test sets using the input TumorGraphGNN model.

        Parameters:
        - model: TumorGraphGNN model to be used for computing the embeddings.
        - train_torch_data: TorchSurvivalDataset object with the training data.
        - test_torch_data: TorchSurvivalDataset object with the test data.
        - train_sorted_keys: list with the ids of the patients in the training set.
        - test_sorted_keys: list with the ids of the patients in the test set.
        - batch_size: integer with the batch size to be used for the dataloaders.
        - device: device to be used for tensor operations.

        Returns:
        - train_embeddings: dataframe with the embeddings for each patient in the training set.
        - test_embeddings: dataframe with the embeddings for each patient in the test set.
        """

        # create train and test dataloaders
        train_dataloader = TrainerTumorModel.get_dataloader(train_torch_data, batch_size=batch_size, shuffle=False)
        test_dataloader = TrainerTumorModel.get_dataloader(test_torch_data, batch_size=batch_size, shuffle=False)

        # compute embeddings for training and test sets
        train_embeddings = TrainerTumorModel.get_embeddings(model, train_dataloader, device=device)
        test_embeddings = TrainerTumorModel.get_embeddings(model, test_dataloader, device=device)

        # create a list of dictionaries with patient ids as keys and embeddings as values for patients in the training set
        train_embeddings_dict = []
        for i, patient_id in enumerate(train_sorted_keys):
            curr_embedding = {'Patient_ID': patient_id}
            for j in range(train_embeddings.shape[1]):
                curr_embedding[f'feature_{j}'] = train_embeddings[i, j].item()
            train_embeddings_dict.append(curr_embedding)
        
        # create a list of dictionaries with patient ids as keys and embeddings as values for patients in the test set
        test_embeddings_dict = []
        for i, patient_id in enumerate(test_sorted_keys):
            curr_embedding = {'Patient_ID': patient_id}
            for j in range(test_embeddings.shape[1]):
                curr_embedding[f'feature_{j}'] = test_embeddings[i, j].item()
            test_embeddings_dict.append(curr_embedding)
        
        # convert the two dictionaries into dataframes and return them
        return pd.DataFrame(train_embeddings_dict), pd.DataFrame(test_embeddings_dict)

    @staticmethod
    def binary_patients_encoding(train_labels_mapping, graphs, patient_ids):
        """
        Encode each TumorGraph in graphs as a binary vector indicating the presence of mutations, using the mutations that appear in the training set.

        Parameters:
        - train_labels_mapping: dictionary that maps the node labels to consecutive integers starting from 0.
        - graphs: list of TumorGraph objects to encode.
        - patient_ids: list with the ids of the patients corresponding to the graphs.
                       It must be aligned with graphs, i.e., patient_ids[i] is the id of the patient corresponding to graphs[i].
        
        Returns:
        - pd.DataFrame(features_dict): pandas dataframe with the encoded features for each patient in the input list of TumorGraph objects and the corresponding patient ids.
        """
        
        # compute the feature matrix for the input graphs
        features = np.zeros((len(graphs), len(train_labels_mapping) - 1), dtype=np.int64)
        for i, graph in enumerate(graphs):
            for node in graph.nodes:
                for label in node.labels:
                    if label != "empty":
                        features[i, train_labels_mapping[label]] = 1

        # create a list of dictionaries with patient ids as keys and features as values for patients in the training set
        features_dict = []
        for i, patient_id in enumerate(patient_ids):
            curr_feature = {'Patient_ID': patient_id}
            for j in range(features.shape[1]):
                curr_feature[f'feature_{j}'] = features[i, j]
            features_dict.append(curr_feature)
        
        return pd.DataFrame(features_dict)

    @staticmethod
    def get_binary_feature_vectors(
        train_phylogenies_path,
        test_phylogenies_path,
        rd_seed=None,
        min_label_occurrences=0,
        ):
        """
        Computes the features for the input training and test sets as binary vectors indicating the presence of mutations.

        Parameters:
        - train_phylogenies_path: string with the path to the file containing the training phylogenies.
        - test_phylogenies_path: string with the path to the file containing the test phylogenies.
        - rd_seed: integer with the random seed for reproducibility.
        - min_label_occurrences: integer with the minimum number of occurrences for a label to be considered.

        Returns:
        - train_features: dataframe with the features for each patient in the training set.
        - test_features: dataframe with the features for each patient in the test set.
        """

        # load the training and test sets with phylogenies
        train_phylogenies = TrainerTumorModel.load_dataset_txt(train_phylogenies_path)
        test_phylogenies = TrainerTumorModel.load_dataset_txt(test_phylogenies_path)

        # create a TumorDataset object that contains patients in the training set sorted by patient id so to allow for reproducibility
        train_sorted_keys = sorted(train_phylogenies.keys())
        train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
        train_data = TumorDataset(train_list_patients)

        # create a TumorDataset object that contains patients in the test set sorted by patient id so to allow for reproducibility
        test_sorted_keys = sorted(test_phylogenies.keys())
        test_list_patients = [test_phylogenies[key] for key in test_sorted_keys]
        test_data = TumorDataset(test_list_patients)

        # compute the set of labels to be considered, based on the number of occurrences in the training set
        if min_label_occurrences > 0:
            train_data.remove_infreq_labels(min_label_occurrences)
        
        # replace the node labels in the test set that are not in the training set with 'empty'
        test_data.replace_label_set(train_data.node_labels(), replace_with='empty')

        # sample one graph per patient
        train_data.sample_one_graph_per_patient(rd_seed=rd_seed)
        test_data.sample_one_graph_per_patient(rd_seed=rd_seed)

        # flatten the two datasets so to have two lists of graphs rather than two lists of patients
        train_graphs = [graph for patient in train_data.dataset for graph in patient]
        test_graphs = [graph for patient in test_data.dataset for graph in patient]

        # map the node labels to consecutive integers starting from 0
        train_labels_mapping = TorchTumorDataset.map_node_labels(train_data.node_labels())
        
        # compute the features for the training and test sets
        train_features = Survival_Features.binary_patients_encoding(train_labels_mapping, train_graphs, train_sorted_keys)
        test_features = Survival_Features.binary_patients_encoding(train_labels_mapping, test_graphs, test_sorted_keys)
        
        return train_features, test_features
