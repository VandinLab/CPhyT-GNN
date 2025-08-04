import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgb
import CloMu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import utils as Utils

class TumorClustering:
    """
    Class with static methods to cluster tumor graphs based on the embeddings extracted from them by a TumorGraphGNN model and evaluate the clustering.
    """

    @staticmethod
    def ball_filter(embeddings, gamma=1):
        """
        Filters the input embeddings by keeping only those in a ball centered in the mean embedding, with radius
        equal to gamma times the mean distance of embeddings from the mean embedding.

        Parameters:
        - embeddings: torch tensor of embeddings.
        - gamma: float representing the proportion of the mean distance from the mean embedding to set the radius of the ball.

        Returns:
        - filtered_embeddings_indices: torch tensor with the indices of the embeddings inside the ball.
        """

        # compute the mean embedding
        mean_embedding = torch.mean(embeddings, dim=0)

        # compute the euclidean distance from the mean embedding for each embedding
        distances = torch.norm(embeddings - mean_embedding, dim=1)

        # set the radius of the ball as some proportion of the mean distance from the mean embedding
        radius = gamma * torch.mean(distances, dim=0)

        # filter the embeddings by keeping only those in the ball centered in the mean embedding
        filtered_embedding_indices = torch.nonzero(distances <= radius).squeeze()

        return filtered_embedding_indices

    @staticmethod
    def percentage_outliers(embeddings, filtered_indices):
        """
        Computes the percentage of embeddings that are considered outliers based on the ball filtering,
        i.e., the percentage of embeddings that are not in the ball centered in the mean embedding.

        Parameters:
        - embeddings: torch tensor with the input embeddings.
        - filtered_indices: torch tensor with the indices of the embeddings inside the computed ball.

        Returns:
        - percentage: float representing the percentage of embeddings that are considered outliers.
        """

        return 100 * (1 - filtered_indices.shape[0] / embeddings.shape[0])

    @staticmethod
    def cluster_embeddings(embeddings, filtered_indices, k_values=range(2, 16), scale=True):
        """
        Computes a k-means clustering of the input embeddings for each of the values of k provided as input.
        The input filtered indices refer to the embeddings to be used for computing the centroids.
        After that, all embeddings are clustered based on the closest centroid.
        The function also prints the average silhouette score for each computed clustering.

        Parameters:
        - embeddings: torch tensor with the input embeddings to be clustered.
        - filtered_indices: torch tensor with the indices of the embeddings to be considered for computing centroids.
        - k_values: array with all clustering sizes to be considered.
        - scale: boolean indicating whether to scale the embeddings before clustering

        Returns:
        - cluster_labels: dictionary with values of k as keys and arrays with cluster labels for embeddings as values.
        """

        # dictionary that will contain the average silhouette of all computed clusterings
        silhouette_clusterings = {}

        # dictionary that will contain the cluster labels for each input value of k
        cluster_labels = {}

        # extract the embeddings to be used for computing the centroids
        all_embeddings = embeddings.cpu().detach().numpy()
        filtered_embeddings = all_embeddings[filtered_indices]

        # scale the embeddings, if needed
        if scale:
            scaler = StandardScaler()
            all_embeddings = scaler.fit_transform(all_embeddings)
            filtered_embeddings = all_embeddings[filtered_indices]

        # compute a clustering for each input value of k
        for k in tqdm(k_values, desc='Computing clusterings', unit='clg'):
            kmeans = KMeans(n_clusters=k, init='k-means++', algorithm='lloyd', n_init=1000, copy_x=True, max_iter=1000)
            kmeans.fit(filtered_embeddings)
            labels = kmeans.predict(all_embeddings)
            cluster_labels[k] = labels
            silhouette_clusterings[k] = silhouette_score(all_embeddings, labels)

        # print the average silhouette score for each computed clustering
        for k, s in silhouette_clusterings.items():
            print(f'K = {k}: s = {s}')
        
        return cluster_labels

    @staticmethod
    def compute_average_silhouette(embeddings, labels):
        """
        Computes the average silhouette score for the input embeddings and cluster labels.

        Parameters:
        - embeddings: torch tensor with the input embeddings.
        - labels: array with the cluster labels for the embeddings. It must be aligned with embeddings, meaning that labels[i] is the cluster label for embeddings[i].

        Returns:
        - silhouette_score: float representing the average silhouette score for the input embeddings and labels.
        """

        return silhouette_score(embeddings.cpu().detach().numpy(), labels)
    
    @staticmethod
    def clusters_sizes(cluster_labels):
        """
        Computes the size of each cluster of a clustering based on the input cluster labels.

        Parameters:
        - cluster_labels: array with the cluster labels for the embeddings.

        Returns:
        - cluster_sizes: dictionary with cluster labels as keys and the number of embeddings in each cluster as values.
        """

        # dictionary that will contain the size of each cluster
        cluster_sizes = {}

        # compute the size of each cluster
        for label in cluster_labels:
            if label not in cluster_sizes:
                cluster_sizes[label] = 0
            cluster_sizes[label] += 1

        return cluster_sizes

    @staticmethod
    def cluster_sizes_all_k(labels):
        """
        Computes the size of each cluster for all computed clusterings based on the input cluster labels.

        Parameters:
        - labels: dictionary with values of k as keys and arrays with cluster labels for embeddings as values.

        Returns:
        - cluster_sizes: dictionary with values of k as keys and dictionaries with cluster labels as keys and the number of embeddings in each cluster as values.
        """

        # dictionary that will contain the size of each cluster for each value of k
        cluster_sizes = {}

        # compute the size of each cluster for each value of k
        for k, labels in labels.items():
            cluster_sizes[k] = TumorClustering.clusters_sizes(labels)

        return cluster_sizes

    @staticmethod
    def print_cluster_sizes_all_k(cluster_sizes):
        """
        Prints the size of each cluster for all computed clusterings.

        Parameters:
        - cluster_sizes: dictionary with values of k as keys and dictionaries with cluster labels as keys and the number of embeddings in each cluster as values.
        """

        # print the size of each cluster for each value of k
        print('\nCluster sizes:\n')
        for k, sizes in cluster_sizes.items():
            print(f'\nK = {k}:\n')
            for label, size in sizes.items():
                print(f'Cluster {label}: {size} embeddings')

    @staticmethod
    def save_cluster_labels(cluster_labels, save_folder):
        """
        Saves the input cluster labels to the specified folder.
        A file for each value of k is created, with the cluster labels for the embeddings.
        Labels are saved in a .npy file such that they are in the same order as the embeddings.

        Parameters:
        - cluster_labels: dictionary with values of k as keys and arrays with cluster labels for embeddings as values.
        - save_folder: string representing the path to the folder where to save the cluster labels.
        """

        # save the cluster labels for each value of k
        for k, labels in cluster_labels.items():
            np.save(os.path.join(save_folder, f'{k}_clusters.npy'), labels)

class Baselines:
    """
    Class with static functions to apply and manage baselines.
    """

    @staticmethod
    def compute_patients_probabilities(patients, trees_probabilities):
        """
        Computes the probability of patients in a dataset, given the probabilities of the single trees belonging to them.
        
        Parameters:
        - patients: array of patients, where each patient consists of some trees.
        - trees_probabilities: list of probabilities assigned to trees in the input data.
        
        Returns:
        - patients_prob: list of probabilities of patients, where element i is the probability of patient i in patients, computed as
                         mean of the probabilities of the trees belonging to that patient.
        """
        
        # list that will contain patients' probabilities
        patients_prob = []

        # index of the first tree of the current patient we are considering
        tree_i = 0

        # iterate over patients and compute, for each patient, its probability as mean of the values associated to its trees
        for patient in patients:
            n_trees_patient = len(patient)
            patients_prob.append(np.mean(trees_probabilities[tree_i:tree_i + n_trees_patient]))
            tree_i += n_trees_patient

        return patients_prob
    
    @staticmethod
    def balanced_CloMu_probabilities_clustering(dataset_path, clusterings_folder, clustering_sizes, test_set, infinite_sites, max_tree_length, iterations):
        """
        Computes a clustering of the input dataset based on CloMu probabilities for each clustering size.

        Parameters:
        - dataset_path: path to the dataset of patients to be clustered.
        - clusterings_folder: path to the folder where to save all computed clusterings.
        - clustering_sizes: number of clusters of the desired clustering.
        - n_iter: number of iterations for each training performed by the function.
        - test_set: ndarray of test patients to be concatenated at the end of each cluster.
        - infinite_sites: whether the infinite sites assumption has to be enabled or not.
        - max_tree_length: maximum length of a tree to be considered, otherwise it is removed. The length of a tree is the number of edges it contains.
        - iterations: number of training epochs for each CloMu instance.
        """

        # iterate through all clustering sizes
        for n_clusters in clustering_sizes:

            # dataset with currently unissegned patients to clusters
            curr_dataset = np.load(dataset_path, allow_pickle=True)
            
            # compute the size of each cluster
            clusters_size = curr_dataset.shape[0] // n_clusters

            # print some information
            print(f'\nComputing clustering with K = {n_clusters}\n')

            # compute the clusters iteratively
            for i in range(n_clusters - 1):

                # print some information
                print(f'Cluster {i + 1}/{n_clusters}\n')
                
                # create all intermediate folders where to save the current clustering, if they do not exist
                os.makedirs(os.path.join(clusterings_folder, f'k_{n_clusters}'), exist_ok=True)

                # save the dataset of still unassigned patients so to be picked by next call of CloMu
                np.save(os.path.join(clusterings_folder, f'k_{n_clusters}', f'temp_dataset.npy'), curr_dataset)

                # train a CloMu instance on the current cluster and apply it to get probabilities for the concatenated test set
                CloMu.trainModel(
                    [os.path.join(clusterings_folder, f'k_{n_clusters}', f'temp_dataset.npy')],                            # path to the current dataset with a cluster as training set and the test set
                    os.path.join(clusterings_folder, f'k_{n_clusters}', f'CloMu_weights_cluster_{i}.pt'),                  # path where to save the trained CloMu weights
                    os.path.join(clusterings_folder, f'k_{n_clusters}', f'CloMu_probabilities_cluster_{i}.npy'),           # path where to save the probabilities assigned both to train and test trees
                    os.path.join(clusterings_folder, f'k_{n_clusters}', f'CloMu_mutations_cluster_{i}.npy'),               # path where to save the mutation names found during training
                    patientNames='',
                    inputFormat='raw',                                                                                     # format of the input dataset
                    infiniteSites=infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                    trainSize='all',                                                                                       # number of patients in the training set for the current cluster
                    maxM=max_tree_length,                                                                                  # maximum length of a tree to be considered, otherwise it is removed            
                    regularizeFactor='default',        
                    iterations=iterations,                                                                                 # number of training epochs
                    verbose=False                                                                                          # whether to print information during training
                )

                # load the probabilities assigned by the trained model to trees in dataset
                probs = np.load(os.path.join(clusterings_folder, f'k_{n_clusters}', f'CloMu_probabilities_cluster_{i}.npy'), allow_pickle=True)
                
                # for each patient, compute its probability as the mean of the probabilities assigned to its trees
                patients_probs = Baselines.compute_patients_probabilities(curr_dataset, probs)

                # find the indices of patients with largest probabilities
                indices = np.argsort(patients_probs)[::-1]
                indices = indices[:clusters_size]

                # assign the patients with high probability to a definitive cluster
                cluster = curr_dataset[indices]

                # save the computed cluster concatenated with test patients
                np.save(os.path.join(clusterings_folder, f'k_{n_clusters}', f'cluster_{i}.npy'), np.concatenate((cluster, test_set), axis=0))

                # remove the found patients from the dataset with patients still to be clustered
                curr_dataset = np.array([curr_dataset[i] for i in range(curr_dataset.shape[0]) if i not in indices], dtype=curr_dataset.dtype)

                # check if we have already reached the final number of cluster and if so, save curr_dataset as last cluster
                if i == n_clusters - 2:
                    np.save(os.path.join(clusterings_folder, f'k_{n_clusters}', f'cluster_{i + 1}.npy'), np.concatenate((curr_dataset, test_set), axis=0))

    @staticmethod 
    def clusters_dimension(clusters_path, n_clusters, test_set):
        """
        Computes the number of patients in each cluster in the input path. Each cluster has a concatenated test set at the end.

        Parameters:
        - clusters_path: path to the folder containing clusters.
        - n_clusters: number of clusters in the folder.
        - test_set: test set of patients appended at the end of each input cluster.

        Returns:
        - clusters_dim: list with the dimension of each cluster in terms of number of patients.
        """

        # number of patients in the test set appended at the end of each cluster
        n_test = test_set.shape[0]

        # list that will contain the number of patients in each cluster
        clusters_dim = []

        # iterate through all clusters
        for i in range(n_clusters):

            # load the current cluster
            curr_cl = np.load(os.path.join(clusters_path, f'cluster_{i}.npy'), allow_pickle=True)

            # compute ad append the number of patients in it, excluding test patients
            clusters_dim.append(curr_cl.shape[0] - n_test)
        
        return clusters_dim

    @staticmethod
    def random_clustering(dataset, cluster_dims):
        """
        Clusters the patients in the input dataset randomly such that the computed clusters have the desired dimensions provided as input.

        Parameters:
        - dataset: dataset of patients to be clustered.
        - cluster_dims: list with the dimension that each cluster must have.

        Returns:
        - clustering: list of random clusters with the desired input dimensions.
        """

        # shuffle the dataset
        shuff_dataset = np.random.permutation(dataset)

        # list that will contain the random clusters to be returned
        clustering = []

        # index of the first patient still unassigned to any cluster
        first_unassigned = 0

        # iterate through cluster dimensions
        for cl_dim in cluster_dims:

            # assign patients to the current cluster
            clustering.append(shuff_dataset[first_unassigned:first_unassigned + cl_dim])

            # update the index of the next patient to be assigned
            first_unassigned += cl_dim

        return clustering

    @staticmethod
    def concatenate_sets(set_1, set_2):
        """
        Concatenates two ndarrays.

        Parameters:
        - set_1: base ndarray.
        - set_2: ndarray to be appended at the end of the base array.

        Returns:
        - concatenation: second ndarray appended after the base one.
        """    
        # array that will be returned
        concatenation = []

        # add the elements of the first array
        for patient in set_1:
            concatenation.append(patient)
        
        # add the elements of the second array
        for patient in set_2:
            concatenation.append(patient)
        
        # convert the list into an ndarray and return it
        return np.array(concatenation, dtype=object)

    @staticmethod
    def probability_bipartition(dataset, probabilities):
        """
        Partitions a dataset of patients into two subsets containing the same number of patients, based on the probability of each patient.

        Parameters:
        - dataset: set of patients to partition.
        - probabilities: probabilities of patients in the input dataset. probabilities[i] is the probabiliy of patient i in the dataset.
        
        Returns:
        - H: patients with high probability;
        - p_h: probabilities of patients in H. p_h[i] is the probability of H[i];
        - L: patients with low probability;
        - p_l: probabilities of patients in L. p_l[i] is the probability of L[i].    
        """

        # list that will contain the first half of patientswith highest probabilities
        H = []
        # list that will contain the probabilities of patients in H
        p_h = []

        # list that will contain half of the patients in the dataset with lowest probabilities
        L = []
        # list that will contain the probabilities of patients in L
        p_l = []

        # half of the total number of patients in the input dataset
        size_l = len(dataset) // 2
        
        # indices of the patients that sort the array of probabilities in non-decreasing order
        sort_indices = np.argsort(probabilities)

        # fill L and p_l
        for i in range(size_l):
            L.append(dataset[sort_indices[i]])
            p_l.append(probabilities[sort_indices[i]])
        
        # fill H and p_h with all other elements
        for i in range(size_l, len(dataset)):
            H.append(dataset[sort_indices[i]])
            p_h.append(probabilities[sort_indices[i]])
        
        # convert all lists in ndarrays and return them
        return np.array(H, dtype=object), np.array(p_h), np.array(L, dtype=object), np.array(p_l)

    @staticmethod
    def move_cluster_elements(A, scores_A, B, scores_B):
        """
        Moves patients from a dataset A to a dataset B if the score of an element in A is closer to the centroid score of B rather than the centroid score of A.

        Parameters:
        - A: dataset from which patients can be moved.
        - scores_A: scores of patients in A.
        - B: dataset to which patients must be moved.
        - scores_B: scores of patients in B.
        
        Returns:
        - new_A: version of the input A without some patients that have been detected to be closer to B. 
        - new_B: version of B with some more patients from A.
        """

        # versions of A and B that will be returned, with some modified patients
        new_A = []
        new_B = []
        
        # compute the centroid scores of the two input clusters
        centr_score_A = np.mean(scores_A)
        centr_score_B = np.mean(scores_B)

        # new_B will contain all patients in B
        for i in range(len(B)):
            new_B.append(B[i])

        # iterate through patients in A
        for i in range(len(A)):

            # if the current patient is closer to A rather than B, append it to new_A
            if np.abs(scores_A[i] - centr_score_B) > np.abs(scores_A[i] - centr_score_A):
                new_A.append(A[i])

            # else, append it to new_B
            else:
                new_B.append(A[i])
        
        # convert the lists into ndarrays and return them
        return np.array(new_A, dtype=object), np.array(new_B, dtype=object)

    @staticmethod
    def unbalanced_CloMu_probabilities_clustering(dataset_path, clusterings_folder, clustering_sizes, test_set, infinite_sites, max_tree_length):
        """
        Computes and saves a clustering of the input dataset for each input clustering size.

        Parameters:
        - dataset_path: path to the dataset with patients to be clustered.
        - clustering_folder: path to the folder where to save all clustering results for this method.
        - clustering_sizes: list with different clustering sizes.
        - test_set: test set of patients not included in clustering to be appended at the end of each computed cluster.
        - infinite_sites: whether the infinite sites assumption has to be enabled or not.
        - max_tree_length: maximum length of a tree to be considered, otherwise it is removed. The length of a tree is the number of edges it contains.
        """
        
        # iterate through all clustering sizes
        for n_clusters in clustering_sizes:

            # dataset with currently unissegned patients to clusters
            curr_dataset = np.load(dataset_path, allow_pickle=True)

            # print some information
            print(f'\nComputing clustering with K = {n_clusters}\n')
            
            # iteratively find a new cluster
            for i in range(n_clusters - 1):
                
                # print some information
                print(f'\nCluster {i + 1}/{n_clusters}\n')
                
                # create all intermediate folders where to save the current clustering, if they do not exist
                os.makedirs(os.path.join(clusterings_folder, f'{n_clusters}_clusters'), exist_ok=True)

                # save the dataset of still unassigned patients so to be picked by next call of CloMu
                np.save(os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'temp_dataset.npy'), curr_dataset)

                # train a CloMu instance on the current cluster and apply it to get probabilities for the concatenated test set
                CloMu.trainModel(
                    [os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'temp_dataset.npy')],                     # path to the current dataset with a cluster as training set and the test set
                    os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'CloMu_weights_cluster_{i}.pt'),           # path where to save the trained CloMu weights
                    os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'CloMu_probabilities_cluster_{i}.npy'),    # path where to save the probabilities assigned both to train and test trees
                    os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'CloMu_mutations_cluster_{i}.npy'),        # path where to save the mutation names found during training
                    patientNames='',
                    inputFormat='raw',                                                                                     # format of the input dataset
                    infiniteSites=infinite_sites,                                                                          # whether the infinite sites assumption has to be enabled or not
                    trainSize='all',                                                                                       # number of patients in the training set for the current cluster
                    maxM=max_tree_length,                                                                                  # maximum length of a tree to be considered, otherwise it is removed            
                    regularizeFactor='default',        
                    iterations=1000,                                                                                       # number of training epochs
                    verbose=False                                                                                          # whether to print information during training
                )

                # load the probabilities assigned by the trained model
                initial_prob = np.load(os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'CloMu_probabilities_cluster_{i}.npy'), allow_pickle=True)
                
                # for each patient, compute its probability as the mean of the probabilities assigned to its trees
                patients_prob = Baselines.compute_patients_probabilities(curr_dataset, initial_prob)

                # divide patients and related probabilities into two equally-sized clusters based on patients' probabilities
                H, prob_h, L, prob_l = Baselines.probability_bipartition(curr_dataset, patients_prob)

                # save the dataset with high probabilities
                np.save(os.path.join(clusterings_folder, f'{n_clusters}_clusters', "initial_H.npy"), H)

                # set the parameters and train an instance of CloMu on H
                input_files = [os.path.join(clusterings_folder, f'{n_clusters}_clusters', "initial_H.npy")]
                model_file = os.path.join(clusterings_folder, f'{n_clusters}_clusters', "H_model.pt")
                prob_file = os.path.join(clusterings_folder, f'{n_clusters}_clusters', "H_model_probabilities.npy")
                mutation_file = os.path.join(clusterings_folder, f'{n_clusters}_clusters', "H_model_mutation_names.npy")

                CloMu.trainModel(
                    input_files,
                    model_file,
                    prob_file,
                    mutation_file,
                    patientNames='',
                    inputFormat='raw',                                                                                    
                    infiniteSites=infinite_sites,                                                                                    
                    trainSize='all',                                                                                      
                    maxM=9,                                                                                                  
                    regularizeFactor='default',        
                    iterations=1000,                                                                                    
                    verbose=False    
                )

                # load the probabilities computed for H by the model just trained
                new_prob_H = np.load(prob_file, allow_pickle=True)

                # compute the new probabilities for patients in H
                new_patients_prob_H = Baselines.compute_patients_probabilities(H, new_prob_H)

                # move from A to B those patients in A that are closer to patients in B in terms of probability
                new_H, new_L = Baselines.move_cluster_elements(H, new_patients_prob_H, L, prob_l)
                
                # save the definitive cluster found in the current iteration with the test set concatenated at the end
                np.save(os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'cluster_{i}.npy'), Baselines.concatenate_sets(new_H, test_set))

                # check if we have already reached the final number of cluster and if so, save new_H as last cluster
                if i == n_clusters - 2:
                    np.save(os.path.join(clusterings_folder, f'{n_clusters}_clusters', f'cluster_{i + 1}.npy'), Baselines.concatenate_sets(new_L, test_set))

                # set the dataset to be considered in the next iteration to new_L
                curr_dataset = new_L

class CloMuClusters:
    """
    Class to create a clustered version of an input dataset in format required by CloMu.
    More specifically, each cluster of patients is saved in a separate .npy file and all clusters have the same test set appended at the end of the cluster so to
    be used for testing with CloMu.
    """

    @staticmethod
    def concatenate_arrays(arr_1, arr_2):
        """
        Concatenates two numpy arrays.

        Parameters:
        - arr_1: base numpy array.
        - arr_2: numpy array to be appended at the end of the base array.

        Returns:
        - concatenation: numpy array with the second input numpy array appended after the base one.
        """

        # array that will be returned
        concatenation = []

        # add the elements of the first array
        for patient in arr_1:
            concatenation.append(patient)
        
        # add the elements of the second array
        for patient in arr_2:
            concatenation.append(patient)
        
        # convert the list into an ndarray and return it
        return np.array(concatenation, dtype=object)

    @staticmethod
    def cluster_patients(dataset, labels):
        """
        Creates a dictionary with clusters, given a clustering of the input dataset specified by the input labels.

        Parameters:
        - dataset: numpy array representing the input dataset that has been clustered.
                   It is an array of patients. Each patient is a list of graphs, each graph is a list of edges.
        - labels: numpy array with cluster labels specifying the cluster of each patient.
                  More specifically, labels[i] is the cluster for patient dataset[i].

        Returns:
        - clusters: dictionary representing a clustering. Cluster labels are ids and each entry is a numpy array with the patients belonging to that cluster.
        """

        # different cluster labels
        cluster_ids = np.unique(labels)

        # initialize each cluster to an empty list of patients
        clusters = {cl_id: [] for cl_id in cluster_ids}
        
        # iterate through patients and assign them to cluster specified by the corresponding label
        for i in range(dataset.shape[0]):
            clusters[labels[i]].append(dataset[i])
        
        # convert each list into a numpy array so to preserve the type of the original dataset
        for key in clusters.keys():
            clusters[key] = np.array(clusters[key], dtype=dataset.dtype)
        
        return clusters

    @staticmethod
    def save_clusters_and_test(clusters, test_set, save_path):
        """
        Saves in a .npy file each cluster with test patients appended in the end.

        Parameters:
        - clusters: dictionary of clusters. Each cluster is a numpy array of patients.
        - test_set: numpy array of test patients to be concatenated at the end of each cluster. Each patient is a list of graphs, each graph is a list of edges.
        - save_path: path to the folder where to save clusters. Cluster i in the clustering is saved as 'cluster_i.npy'.
        """

        # create the folders in the intermediate path where to save the clusters, if they do not exist
        os.makedirs(save_path, exist_ok=True)

        # iterate through all clusters and save each of them in a different file, with the test set appended at the end
        for label, cluster in clusters.items():
            concat_cluster = CloMuClusters.concatenate_arrays(cluster, test_set)
            np.save(os.path.join(save_path, f'cluster_{label}.npy'), concat_cluster)

    @staticmethod
    def RECAP_extract_cluster_indices(path):
        """
        Extracts the cluster labels of patients from the .txt file that results from the application of RECAP.
        
        Parameters:
        - path: path to the output .txt file from the application of RECAP.

        Returns:
        - indices_list: list of cluster indices, where indices_list[i] is the cluster index for patient i in the original dataset provided as input to RECAP.
        """

        # list of cluster indices for patients
        indices_list = []
        
        # open the input file
        with open(path, 'r') as file:
            
            # iterate over the lines of the file
            for line in file:
                
                # check whether the line contains the word "cluster" and extract the index in case
                if 'cluster' in line:
                    
                    # the cluster index is always the first word in the line
                    indices_list.append(line[0])
        
        # remove the first value in the list, because it is the total number of clusters
        indices_list = indices_list[1:]

        return indices_list

class Ensemble:
    """
    Class with static methods to train and combine CloMu instances on different clusters using an ensemble learning approach.
    """

    @staticmethod
    def train_on_clusters(clusters_path, weights_path, probabilities_path, mutations_path, test_set_path, n_clusters, input_format, infinite_sites, max_tree_length, regularize_factor, epochs):
        """
        Trains a different instance of CloMu on each cluster of a clustering obtained with an input clustering method.

        Parameters:
        - clusters_path: path to the folder with clusters.
        - weights_path: path to the folder where to save CloMu weights for the clusters.
        - probabilities_path: path to the folder where to save CloMu probabilities for the clusters.
        - mutations_path: path to the folder where to save CloMu mutations for the clusters.
        - test_set_path: path to the test set concatenated to all clusters.
        - n_clusters: number of clusters.
        - input_format: format in which each cluster is saved,
        - infinite_sites: whether the infinite sites assumption must be enabled during training or not.
        - max_tree_length: maximum length that an input tree can have.
        - regularize_factor: regularize factor for CloMu.
        - epochs: number of epochs for which each cluster must be trained.
        """

        # create the intermediate folders in the paths where to save results, if they do not exist
        os.makedirs(weights_path, exist_ok=True)
        os.makedirs(probabilities_path, exist_ok=True)
        os.makedirs(mutations_path, exist_ok=True)

        # load the test set concatenated to all clusters
        test_set = np.load(test_set_path, allow_pickle=True)

        # number of patients in the test set concatenated to all clusters
        n_test_patients = test_set.shape[0]

        # print information about the current clustering
        print(f'\nTraining on clustering with K = {n_clusters}:\n')

        # iterate through all clusters
        for i in range(n_clusters):

            # print information about training on the current cluster
            print(f'\nTraining on cluster {i + 1}/{n_clusters}:\n')

            # number of patients in the current cluster, excluding test patients
            curr_n_train_patients = np.load(os.path.join(clusters_path, f'cluster_{i}.npy'), allow_pickle=True).shape[0] - n_test_patients

            # train a CloMu instance on the current cluster and apply it to get probabilities for the concatenated test set
            CloMu.trainModel(
                [os.path.join(clusters_path, f'cluster_{i}.npy')],                                        # path to the current dataset with a cluster as training set and the test set
                os.path.join(weights_path, f'CloMu_weights_cluster_{i}.pth'),                             # path where to save the trained CloMu weights
                os.path.join(probabilities_path, f'CloMu_probabilities_cluster_{i}.npy'),                 # path where to save the probabilities assigned both to train and test trees
                os.path.join(mutations_path, f'CloMu_mutations_cluster_{i}.npy'),                         # path where to save the mutation names found during training
                patientNames='',
                inputFormat=input_format,                                                                 # format of the input dataset
                infiniteSites=infinite_sites,                                                             # whether the infinite sites assumption has to be enabled or not
                trainSize=curr_n_train_patients,                                                          # number of patients in the training set for the current cluster
                maxM=max_tree_length,                                                                     # maximum length of a tree to be considered, otherwise it is removed            
                regularizeFactor=regularize_factor,                                                       # regularize factor
                iterations=epochs,                                                                        # number of training epochs
                verbose=False                                                                             # whether to print information during training
            )

    @staticmethod
    def ensemble_of_models(prob_arrays):
        """
        Returns the final score of each patient, given the scores assigned to it by all models in the ensemble.
        The returned score is the maximum among the probabilities assigned by all models.
        In case CloMu assigned nan as score, it is not considered. This happens sometimes if CloMu is trained on
        a single tree.

        Parameters:
        - prob_arrays: list of arrays of probabilities. Each array contains the output probabilities from a given
                       model in the considered ensemble.

        Returns:
        - scores: array with final scores for the input patients.
        """

        # number of patients for which probabilities have been computed
        n_patients = len(prob_arrays[0])

        # array that will contain the ensemble score for each patient
        scores = []

        # the ensemble score for a patient is the maximum among the probabilities assigned by all models
        for i in range(n_patients):
            scores.append(np.max([array[i] for array in prob_arrays if not np.isnan(array[i])]))
        
        # convert the list into an array and return it
        return np.array(scores)

    @staticmethod
    def get_clustering_patients_probabilities(n_clusters, test_set, folder_path):
        """
        Packs the probabilities assigned by the models trained on all clusters into a list of numpy arrays with an numpy array for each cluster.
        Moreover, it computes patient probabilities from the probabilities assigned to their trees as their mean.

        Parameters:
        - n_clusters: number of computed clusters.
        - test_set: test set concatenated at the end of each cluster.
        - folder_path: path to the folder containing a file with probabilities for each cluster.

        Returns:
        - list_arrays: list of numpy arrays. The list contains an ndarray of probabilities for each cluster.
                       each entry of an ndarray is the probability assigned by the model trained on the
                       related cluster to a test patient.
        """

        # number of trees in the test set
        n_test_trees = Utils.get_n_items_list_of_lists(test_set)

        # list that will contain an ndarray of probabilities for each cluster
        list_arrays = []

        # load the probabilities assigned by the different models to the test set and compute patient probabilities
        for i in range(n_clusters):
            curr_prob_path = os.path.join(folder_path, f'CloMu_probabilities_cluster_{i}.npy')
            curr_prob = np.load(curr_prob_path, allow_pickle=True)
            test_trees_probs = curr_prob[-n_test_trees:]
            list_arrays.append(Ensemble.test_patients_probabilities(test_set, test_trees_probs))
        
        return list_arrays

    @staticmethod
    def test_patients_probabilities(test_set, test_trees_probs):
        """
        Computes the probability assigned to each patient in the test set based on the probabilities of its trees.

        Parameters:
        - test_set: array of patients, where each patient contains a list of trees.
        - test_trees_probs: array of probabilities assigned to each tree in the test set.

        Returns:
        - patients_probs: array of probabilities assigned to test patient, computed as the mean probability of their trees.
        """

        # list that will contain the probabilities assigned to test patients
        patients_probs = []

        # index of the next tree to be considered
        index_next_tree = 0

        # iterate through all patients and compute the probability of a patient as the mean of the probabilities of its trees
        for patient in test_set:
            n_trees_curr_patient = len(patient)
            curr_patient_prob = np.mean(test_trees_probs[index_next_tree:index_next_tree + n_trees_curr_patient])
            patients_probs.append(curr_patient_prob)
            index_next_tree += n_trees_curr_patient

        # convert the list into an array and return it
        return np.array(patients_probs)

    @staticmethod
    def all_percentages(results_dir, clustering_sizes, baseline_methods):
        """
        Computes the percentage of test trees to which the method to compare assigns a larger score w.r.t. all other input methods across all random seeds and all clustering sizes.

        Parameters:
        - results_dir: path to the folder containing the subdirectories for different random seeds, containing the results produced by the cancer progression modeling experiments.
        - clustering_sizes: list of clustering sizes to be considered.
        - baseline_methods: list of clustering methods to be compared with the GNN-based method.

        Returns:
        - percentages: dictionary with all percentage values.
                        percentages[method][n_clusters][rd_seed] is the percentage of test trees in which method is beated by the GNN-based method with the chosen random seed and
                        number of clusters.
        """

        # method to be compared against all the others
        method_to_compare = 'GNN'

        # standard CloMu method
        standard_CloMu_method = 'Standard_CloMu'

        # dictionary that will contain all percentages across experiments
        percentages = {}

        # consider also baseline CloMu, which is not included among the input methods
        percentages['baseline_CloMu'] = {}

        # consider all clustering sizes
        for n_clusters in clustering_sizes:

            # dictionary for the current number of clusters
            percentages['baseline_CloMu'][n_clusters] = {}

            # iterate through all subdirectories: there is one for each random seed
            for rd_seed_dir in Utils.listdir_nohidden(results_dir):

                # get the random seed from the name of the subdirectory
                rd = int(rd_seed_dir.split('_')[-1])

                # load the test set constructed with the current random seed
                test_set = np.load(os.path.join(results_dir, rd_seed_dir, 'pre_processed_data', 'test_set.npy'), allow_pickle=True)

                # compute the number of trees in the test set
                n_trees_test = Utils.get_n_items_list_of_lists(test_set)

                # path to the folder containing the clustering for the GNN-based method with current size for current random seed
                clustering_folder_base = os.path.join(results_dir, rd_seed_dir, method_to_compare, f'k_{n_clusters}')

                # scores assigned to test trees by the method to compare with current random seed and clustering size
                scores_base = Ensemble.ensemble_of_models(Ensemble.get_clustering_patients_probabilities(n_clusters, test_set, clustering_folder_base))

                # scores assigned by baseline CloMu to test trees with current random seed
                scores_baseline_CloMu = np.load(os.path.join(results_dir, rd_seed_dir, standard_CloMu_method, f'Standard_CloMu_probabilities.npy'), allow_pickle=True)[-n_trees_test:]

                # scores assigned by baseline CloMu to test patients
                scores_baseline_CloMu = Ensemble.test_patients_probabilities(test_set, scores_baseline_CloMu)

                # percentage of cases in which the model to compare beats the baseline CloMu model
                percentages['baseline_CloMu'][n_clusters][rd] = np.sum(scores_base >= scores_baseline_CloMu) / scores_base.shape[0] * 100

        # now iterate through all input clustering methods
        for method in baseline_methods:

            # the method to compare must not be compared w.r.t. itself
            if method != method_to_compare:

                # dictionary for the current method
                percentages[method] = {}

                # consider now all clustering sizes
                for n_clusters in clustering_sizes:

                        # dictionary for the current number of clusters and clustering method
                        percentages[method][n_clusters] = {}

                        # iterate through all subdirectories: there is one for each random seed
                        for rd_seed_dir in Utils.listdir_nohidden(results_dir):
                            
                            # get the random seed from the name of the subdirectory
                            rd = int(rd_seed_dir.split('_')[-1])

                            # load the test set constructed with the current random seed
                            test_set = np.load(os.path.join(results_dir, rd_seed_dir, 'pre_processed_data', 'test_set.npy'), allow_pickle=True)

                            # compute the number of trees in the test set
                            n_trees_test = Utils.get_n_items_list_of_lists(test_set)

                            # path to the folder containing the clustering for the GNN-based method with current size for current random seed
                            clustering_folder_base = os.path.join(results_dir, rd_seed_dir, method_to_compare, f'k_{n_clusters}')

                            # scores assigned to test trees by the method to compare with current random seed and clustering size
                            scores_base = Ensemble.ensemble_of_models(Ensemble.get_clustering_patients_probabilities(n_clusters, test_set, clustering_folder_base))

                            # path to the folder containing the clustering with current size for current random seed and current method
                            clustering_folder_curr_method = os.path.join(results_dir, rd_seed_dir, method, f'k_{n_clusters}')

                            # scores assigned to test trees by the current method with current random seed and clustering size
                            scores_curr_method = Ensemble.ensemble_of_models(Ensemble.get_clustering_patients_probabilities(n_clusters, test_set, clustering_folder_curr_method))

                            # percentage of cases in which the model to compare beats the current model
                            percentages[method][n_clusters][rd] = np.sum(scores_base >= scores_curr_method) / scores_base.shape[0] * 100

        return percentages

    @staticmethod
    def percentages_same_k(percentages):
        """
        Computes the mean percentage of times each method is beaten across different random seeds for each clustering size.

        Parameters:
        - percentages: dictionary with percentage values.
                    percentages[method][n_clusters][rd_seed] is the percentage of test trees in which method is beaten by the method to compare
                    with the chosen random seed and number of clusters.

        Returns:
        - mean_rd_perc: dictionary with mean percentages across different random seeds for all methods and clustering sizes.
                        mean_rd_perc[method][n_clusters] is the mean percentage of times the method is beaten for a given number of clusters.
        """

        # information about what will be printed
        print('\nAvg number of times each method is beated across random seeds:')

        # dictionary that will contain mean percentages across different random seeds for all methods
        mean_rd_perc = {}

        # iterate through methods
        for method in percentages.keys():

            # separator for better readability
            print('\n')
            
            # dictionary that will contain mean percentages across different random seeds for the current method
            mean_rd_perc[method] = {}

            # iterate through clustering sizes
            for n_clusters in percentages[method].keys():

                # mean of percentages across different random seeds for current method and number of clusters
                mean_rd_perc[method][n_clusters] = np.mean([perc for rd, perc in percentages[method][n_clusters].items()])

                # print the computed statistics
                print(f'{method}, K = {n_clusters}: {mean_rd_perc[method][n_clusters]: .2f}%')

        return mean_rd_perc

    @staticmethod
    def percentages_same_method(perc_same_k):
        """
        Computes the mean percentage of times each method is beaten across different random seeds and clustering sizes.

        Parameters:
        - perc_same_k: dictionary with mean percentages across different random seeds for all methods and clustering sizes.
                    perc_same_k[method][n_clusters] is the mean percentage of times the method is beaten for a given number of clusters.

        Returns:
        - mean_rd_perc: dictionary with mean percentages across different random seeds for all methods.
                        mean_rd_perc[method] is the mean percentage of times the method is beaten across all clustering sizes and random seeds.
        """

        # information about what will be printed
        print('\nAvg number of times each method is beated:\n')

        # dictionary that will contain mean percentages across different random seeds and clustering sizes for all methods
        mean_rd_perc = {}

        # iterate through all compared methods
        for method in perc_same_k.keys():
            
            # mean times across different random seeds and clustering sizes in which the current method is beaten
            mean_rd_perc[method] = np.mean([perc for n_clusters, perc in perc_same_k[method].items()])

            # print the computed mean for the current method
            print(f'{method}: {mean_rd_perc[method]: .2f}%')

    @staticmethod
    def plot_probabilities(probabilities, label):
        """
        Plots an array of probabilities, after having sorted it in non-increasing order.

        Parameters:
        - probabilities: array with a probability for each patient.
        - label: label for the plotted curve to be inserted in the legend.
        """

        # plot the sorted probabilities
        sns.set_theme()

        sns.lineplot(data=np.log10(np.sort(probabilities)[::-1]), label=label)

        plt.title('Sorted probabilities')
        plt.xlabel('patient')
        plt.ylabel('log_10(Pr)')

        plt.show()

    @staticmethod
    def plot_multiple_probability_arrays(baseline_probabilities, prob_arrays):
        """
        Plots the array of probabilities from the baseline CloMu model and a curve for each ndarray of probabilities in the same plot.
        The ndarrays must be aligned, i.e., entries in the same position in different ndarrays must be referred to the same patient.

        Parameters:
        - baseline_probabilities: array with baseline probabilities from a single trained CloMu model.
        - prob_arrays: dictionary of aligned ndarrays of probabilities with clustering methods as keys.
        """

        # set the sns theme for plotting
        sns.set_theme('paper')

        # plot the baseline probabilities from a single CloMu model
        sns.lineplot(data=np.log10(np.sort(baseline_probabilities)[::-1]), label='baseline CloMu')

        # plot each sorted array in the same figure
        for method, probs in prob_arrays.items():
            sns.lineplot(data=np.log10(np.sort(probs)[::-1]), label=f'{method}')

        plt.title('Sorted probabilities')
        plt.xlabel('patient')
        plt.ylabel('log_10(Pr)')

        plt.show()

    @staticmethod
    def plot_corresponding_patient_probs(baseline_probabilities, prob_arrays):
        """
        Plots the array of probabilities from the baseline CloMu model and a curve for each ndarray of probabilities in the same plot.
        The difference with respect to plot_multiple_probability_arrays is that here the test patients are sorted by baseline probability
        and then all other ndarrays are plotted with corresponding patients and not sorted.
        That is, points with the same coordinate x refer to the scores assigned by different models to the same test patient x.
        The ndarrays must be aligned, i.e., entries in the same position in different ndarrays must be referred to the same patient.

        Parameters:
        - baseline_probabilities: array with baseline probabilities from a single trained CloMu model.
        - prob_arrays: dictionary of aligned ndarrays of probabilities with clustering methods as keys.
        """

        # find the patient indices that sort the baseline probabilities in non-increasing order
        ordered_patients = np.argsort(baseline_probabilities)[::-1]

        # set the sns theme for plotting
        sns.set_theme('paper')

        # plot the ordered baseline probabilities from a single CloMu model
        sns.lineplot(data = np.log10(baseline_probabilities[ordered_patients]), label='baseline CloMu', color='black')

        # plot each array with aligned patients in the same figure
        for method, probs in prob_arrays.items():
            sns.scatterplot(data=np.log10(probs[ordered_patients]), label=f'{method}')

        plt.title('Sorted probabilities')
        plt.xlabel('patient')
        plt.ylabel('log_10(Pr)')

        plt.show()

    @staticmethod
    def compute_test_probabilities(folder, methods, num_clusters, test_set):
        """
        Returns the probabilities assigned to test patients by the ensembles based on each clustering method among those provided as input.

        Parameters:
        - folder: path to the folder containing a subfolder for each clustering method, where probabilities for all clusterings are contained.
        - methods: list of strings specifying the clustering methods to be considered.
        - num_clusters: size of the clustering to be considered for the input methods.
        - n_trees_test: number of trees in the test set.

        Returns:
        - test_probs: dictionary of aligned ndarrays, one for each input clustering method. The keys are the input string ids for clustering methods.
        """

        # dictionary that will contain clustering method string ids as keys and ndarrays of test probabilities as values
        test_probs = {}

        # iterate through the input clustering methods
        for cl_method in methods:

            # path to the folder with probabilities for the current clustering method and input clustering size
            method_folder = os.path.join(folder, cl_method, f'{num_clusters}_clusters')

            # get the probabilities assigned to test patients by all clusters obtained with the current clustering method and clustering size
            probs_clusters = Ensemble.get_clustering_patients_probabilities(num_clusters, test_set, method_folder)

            # extract the ensemble scores for test patients
            test_scores = Ensemble.ensemble_of_models(probs_clusters)

            # add the scores for the current clustering method to the dictionary
            test_probs[cl_method] = test_scores

        return test_probs

    @staticmethod
    def percentage_comparison(baseline, methods_scores):
        """
        Computes and prints some percentage comparisons between a baseline method and all other methods provided as input.

        Parameters:
        - baseline: string id that identifies the method that must be compared against all other methods. It must be one of the keys in methods_scores.
        - methods_scores: dictionary with a string id referring to the method used to obtain such scores as keys and aligned scores as values.
        """

        # scores for the baseline
        base_scores = methods_scores[baseline]

        # dictionary that will contain a percentage for each comparison
        percentages = {}

        # compute the percentage of patients for which the baseline score is larger than the score assigned by the each method
        for method in methods_scores.keys():
            if method != baseline:
                percentages[method] = np.sum(base_scores >= methods_scores[method]) / len(base_scores) * 100
                print(f'{baseline} score larger than {method} score: {percentages[method]: .2f}%')

    @staticmethod
    def print_stats(methods_scores):
        """
        Computes and prints some statistics about the scores obtained by all other methods provided as input.

        Parameters:
        - methods_scores: dictionary with a string id referring to the method used to obtain such scores as keys and aligned scores as values.
        """

        # iterate through all input methods
        for method in methods_scores.keys():

            # compute and print some statistics for the current model
            print(method)
            print(f'Mean: {np.log10(np.mean(methods_scores[method]))}')
            print(f'Median: {np.log10(np.median(methods_scores[method]))}')
            print(f'Max: {np.log10(np.max(methods_scores[method]))}')
            print(f'Min: {np.log10(np.min(methods_scores[method]))}\n')

    @staticmethod
    def make_tidy(percentages):
        """
        Creates a tidy version of the input dictionary with percentages.
        That is, each row of the resulting dataframe will contain a percentage for a given method, number of clusters and random seed.

        Parameters:
        - percentages: dictionary with all percentage values.
                       percentages[method][n_clusters][rd_seed] is the percentage of test patients in which method is beated by the method to compare with the chosen random seed and
                       number of clusters.
        
        Returns:
        - tidy_df: dataframe with a row for each percentage value and columns "method", "n_clusters" and "experiment_id".
        """

        # list that will contain an entry for each observation
        tidy_list = []

        # iterate through all items in the input nested dictionary so to fill the list
        for method in percentages.keys():
            for n_clusters in percentages[method].keys():
                for rd_seed, perc in percentages[method][n_clusters].items():
                    tidy_list.append({'method': method, 'n_clusters': n_clusters, 'experiment_id': rd_seed, 'percentage': perc})
        
        # convert the list into a dataframe and return it
        return pd.DataFrame(tidy_list)

    @staticmethod
    def compute_all_test_scores(results_dir, clustering_sizes, clustering_methods):
        """
        Computes the test scores assigned by all methods to test patients for all random seeds and clustering sizes.
        
        Parameters:
        - results_dir: path to the folder containing the subdirectories for different random seeds, containing the results produced by the cancer progression modeling experiments.
        - clustering_sizes: list of clustering sizes.
        - clustering methods: list of string ids referring to the clustering methods to be considered.

        Returns:
        - test_scores: dictionary with all test scores.
                       test_scores[method][n_clusters][rd_seed][i] is the test score assigned by method with the chosen random seed and number of clusters to the i-th test patient.
        """

        # standard CloMu method
        standard_CloMu_method = 'Standard_CloMu'

        # dictionary that will contain all test scores
        test_scores = {}

        # fill the dictionary with test scores by the baseline CloMu model
        test_scores['baseline_CloMu'] = {}
        for n_clusters in clustering_sizes:
            test_scores['baseline_CloMu'][n_clusters] = {}
            for rd_seed_dir in Utils.listdir_nohidden(results_dir):
                rd = int(rd_seed_dir.split('_')[-1])
                test_set = np.load(os.path.join(results_dir, rd_seed_dir, 'pre_processed_data', 'test_set.npy'), allow_pickle=True)
                n_trees_test = Utils.get_n_items_list_of_lists(test_set)
                scores_baseline_CloMu = np.load(os.path.join(results_dir, rd_seed_dir, standard_CloMu_method, f'Standard_CloMu_probabilities.npy'), allow_pickle=True)[-n_trees_test:]
                test_scores['baseline_CloMu'][n_clusters][rd] = Ensemble.test_patients_probabilities(test_set, scores_baseline_CloMu)

        # now iterate through all input clustering methods and fill the dictionary with test scores
        for method in clustering_methods:
            test_scores[method] = {}
            for n_clusters in clustering_sizes:
                test_scores[method][n_clusters] = {}
                for rd_seed_dir in Utils.listdir_nohidden(results_dir):
                    rd = int(rd_seed_dir.split('_')[-1])
                    test_set = np.load(os.path.join(results_dir, rd_seed_dir, 'pre_processed_data', 'test_set.npy'), allow_pickle=True)
                    n_trees_test = Utils.get_n_items_list_of_lists(test_set)
                    clustering_folder_curr_method = os.path.join(results_dir, rd_seed_dir, method, f'k_{n_clusters}')
                    test_scores[method][n_clusters][rd] = Ensemble.ensemble_of_models(Ensemble.get_clustering_patients_probabilities(n_clusters, test_set, clustering_folder_curr_method))

        return test_scores
    
    @staticmethod
    def compute_global_percentages(test_scores):
        """
        Computes the percentage of test patients to whom each method assigns the largest score compared to all other methods, for a given clustering size and random seed.

        Parameters:
        - test_scores: dictionary with all test scores.
                       test_scores[method][n_clusters][rd_seed][i] is the test score assigned by method with the chosen random seed and number of clusters to the i-th test patient.
        
        Returns:
        - global_perc: dictionary with global percentages.
                       global_perc[method][n_clusters][rd_seed] is the percentage of test patients to whom method assigns the largest score compared to all other methods with the chosen random seed and number of clusters.
        """
            
        # dictionary that will contain global percentages
        global_perc = {}

        # iterate through all methods
        for method in test_scores.keys():
            global_perc[method] = {}
            for n_clusters in test_scores[method].keys():
                global_perc[method][n_clusters] = {}
                for rd_seed in test_scores[method][n_clusters].keys():
                    global_perc[method][n_clusters][rd_seed] = 0
                    for i in range(len(test_scores[method][n_clusters][rd_seed])):
                        is_best = True
                        for method_to_compare in test_scores.keys():
                            if method_to_compare != method:
                                if test_scores[method][n_clusters][rd_seed][i] < test_scores[method_to_compare][n_clusters][rd_seed][i]:
                                    is_best = False
                                    break
                        if is_best:
                            global_perc[method][n_clusters][rd_seed] += 1
                    global_perc[method][n_clusters][rd_seed] = global_perc[method][n_clusters][rd_seed] / len(test_scores[method][n_clusters][rd_seed]) * 100

        return global_perc

    @staticmethod
    def plot_tidy_percentages(tidy_percentages_df, method_to_compare, save_path=None):
        """
        Plots the input tidy dataframe with percentage number of test patients to whom method_to_compare assigns a larger score w.r.t. all other methods.
        It also saves the plot in the specified path, if provided.

        Parameters:
        - tidy_percentages_df: dataframe with a row for each experiment and columns "percentage", "method", "n_clusters" and "experiment_id".
        - method_to_compare: string id that identifies the method used as baseline for comparisons. It must be one of the ids in tidy_percentages_df['method'].
        - save_path: path where to save the plot. If None, the plot is not saved.
        """

        # remove the method to compare from the dataframe
        percentages_df = tidy_percentages_df[tidy_percentages_df['method'] != method_to_compare]

        # change the name of the models to be used to label the axes
        new_method_to_compare = method_to_compare.replace('_', ' ')
        names_mapping = {
            'baseline_CloMu': f'{new_method_to_compare}\nvs\nStandard',
            'Random': f'{new_method_to_compare}\nvs\nRandom',
            'CloMu_based': f'{new_method_to_compare}\nvs\nCloMu',
            'Random_100': f'{new_method_to_compare}\nvs\nRandom 100',
            'RECAP': f'{new_method_to_compare}\nvs\nRECAP',
            'RECAP_100': f'{new_method_to_compare}\nvs\nRECAP 100',
            'oncotree2vec': f'{new_method_to_compare}\nvs\nO2V',
            'GNN': f'{new_method_to_compare}\nvs\nGNN',
            'GNN_100': f'{new_method_to_compare}\nvs\nGNN 100'
        }
        percentages_df['method'] = percentages_df['method'].replace(names_mapping)

        # set theme and style
        sns.set_theme('paper')
        sns.set_style()

        # number of methods and number of clusters
        n_methods = percentages_df['method'].nunique()
        n_clusters = percentages_df['n_clusters'].nunique()

        # define different colors for different methods and different luminance values for different clusters
        colors = sns.color_palette('tab10', n_methods)
        luminance_values = np.linspace(0.3, 0.6, n_clusters)[::-1]
        box_colors = []
        for luminance in luminance_values:
            for base_color in colors:
                box_colors.append(sns.set_hls_values(to_rgb(base_color), l=luminance))

        # create figure and axes
        fig, ax = plt.subplots(figsize=(9, 6))

        # plot the boxes
        sns.boxplot(
            data=percentages_df,
            x='method',
            y='percentage',
            hue='n_clusters',
            dodge=True,
            ax=ax
        )

        # apply custom colors to the boxplot
        box_patches = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
        for patch, color in zip(box_patches, box_colors):
            patch.set_facecolor(color)

        # create a custom legend
        handles = []
        for i in range(n_clusters):
            handles.append(tuple(box_patches[i * n_methods : i * n_methods + n_methods]))
        ax.legend(
            handles=handles,
            labels=[f'K={c + 2}' for c in range(n_clusters)],
            title='Clustering',
            handlelength=4,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            fontsize=16,
            title_fontsize=18
        )

        # plot a horizontal line at 50%
        plt.axhline(y=50, color='r', linestyle='--')

        # set title, labels and ticks
        plt.title('Relative Performances, F=100', fontsize=24)
        plt.xlabel('Method', fontsize=20)
        plt.ylabel('Patients (%)', fontsize=20)
        plt.ylim(29, 101)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()

        # remove spines
        sns.despine(offset=10, trim=True)

        # save the plot
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        # show the plot
        plt.show()

    @staticmethod
    def plot_global_percentages(global_percentages_df, save_path=None):
        """
        Plots the input tidy dataframe with percentage number of test patients to whom each method assigns a larger score w.r.t. all other methods.
        It also saves the plot in the specified path, if provided.

        Parameters:
        - global_percentages_df: dataframe with a row for each experiment and columns "percentage", "method", "n_clusters" and "experiment_id".
        - save_path: path where to save the plot. If None, the plot is not saved.
        """

        # change the name of the models to be used to label the axes
        names_mapping = {
            'baseline_CloMu': 'Standard',
            'Random': 'Random',
            'CloMu_based': 'CloMu',
            'Random_100': 'Random 100',
            'RECAP': 'RECAP',
            'RECAP_100': 'RECAP 100',
            'oncotree2vec': 'O2V',
            'GNN': 'GNN',
            'GNN_100': 'GNN 100'
        }
        global_percentages_df['method'] = global_percentages_df['method'].replace(names_mapping)

        # set theme and style
        sns.set_theme('paper')
        sns.set_style()

        # number of methods and number of clusters
        n_methods = global_percentages_df['method'].nunique()
        n_clusters = global_percentages_df['n_clusters'].nunique()

        # define different colors for different methods and different luminance values for different clusters
        colors = sns.color_palette('tab10', n_methods)
        luminance_values = np.linspace(0.3, 0.6, n_clusters)[::-1]
        box_colors = []
        for luminance in luminance_values:
            for base_color in colors:
                box_colors.append(sns.set_hls_values(to_rgb(base_color), l=luminance))

        # create figure and axes
        fig, ax = plt.subplots(figsize=(9, 6))

        # plot the boxes
        sns.boxplot(
            data=global_percentages_df,
            x='method',
            y='percentage',
            hue='n_clusters',
            dodge=True,
            ax=ax
        )

        # apply custom colors to the boxplot
        box_patches = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
        for patch, color in zip(box_patches, box_colors):
            patch.set_facecolor(color)

        # create a custom legend
        handles = []
        for i in range(n_clusters):
            handles.append(tuple(box_patches[i * n_methods : i * n_methods + n_methods]))
        ax.legend(
            handles=handles,
            labels=[f'K={c + 2}' for c in range(n_clusters)],
            title='Clustering',
            handlelength=4,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            fontsize=16,
            title_fontsize=18
        )

        # set title, labels and ticks
        plt.title('Absolute Performances, F=100', fontsize=24)
        plt.xlabel('Method', fontsize=20)
        plt.ylabel('Patients (%)', fontsize=20)
        plt.ylim(-1, 61)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.tight_layout()

        # remove spines
        sns.despine(offset=10, trim=True)

        # save the plot
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        # show the plot
        plt.show()