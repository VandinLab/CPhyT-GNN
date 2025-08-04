import os
import argparse
import torch
import pandas as pd
from tumor_model import TumorDataset
from tumor_model import TorchTumorDataset
from tumor_model import TumorGraphGNN
from tumor_model import TrainerTumorModel as Trainer
import utils as Utils

def parse_args():
    '''
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    '''

    # create the argument parser
    parser = argparse.ArgumentParser(description="Compute unsupervised embeddings for an input dataset of phylogenetic trees, using a trained model")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--train_set', type=str, required=True, help='Path to the .txt file containing the dataset with the phylogenetic trees used for training the model')
    required.add_argument('-i', '--trees_to_embed', type=str, required=True, help='Path to the .txt file containing the dataset with the phylogenetic trees to embed')
    required.add_argument('-w', '--weights', type=str, required=True, help='Path to the file storing the weights of the trained model')
    required.add_argument('-o', '--embeddings', type=str, required=True, help='Path to the .csv file where to save the embeddings')

    # optional arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=8, help='Max number of CPU cores for PyTorch')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the training set to be considered. It must coincide with the value used for training')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation". It must coincide with the encoding used for training')
    parser.add_argument('--h_1', type=int, default=64, help='Output size of the first GCN hidden layer. It must coincide with the value used for training')
    parser.add_argument('--h_2', type=int, default=64, help='Output size of the second GCN hidden layer. It must coincide with the value used for training')
    parser.add_argument('-d', '--embedding_dim', type=int, default=64, help='Embedding dimension. It must coincide with the value used for training')
    parser.add_argument('--dropout_prob_1', type=float, default=0.3, help='Dropout probability before the second GCN layer. It must coincide with the value used for training')
    parser.add_argument('--dropout_prob_2', type=float, default=0.3, help='Dropout probability before the final linear layer. It must coincide with the value used for training')
    parser.add_argument('--batch_normalization', action='store_true', help='Use batch normalization. It must coincide with the value used for training')    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the dataloader with the trees to be embedded')
    
    return parser.parse_args()

if __name__ == '__main__':

    # parse the command line arguments
    args = parse_args()
    
    # set the device to use for tensor operations
    device = Utils.get_device(args.device)
    print(f"Using device: {device}")

    # limit the cores used by torch
    torch.set_num_threads(args.max_n_cores)
    torch.set_num_interop_threads(args.max_n_cores)

    # load the training and set with phylogenies to be embedded
    train_phylogenies = Trainer.load_dataset_txt(args.train_set)
    to_embed_phylogenies = Trainer.load_dataset_txt(args.trees_to_embed)

    # create a TumorDataset object for the training set that contains patients sorted by patient id so to allow for reproducibility
    train_sorted_keys = sorted(train_phylogenies.keys())
    train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
    train_data = TumorDataset(train_list_patients)

    # create a TumorDataset object for the phylogenies to be embedded that contains patients sorted by patient id so to allow for reproducibility
    to_embed_sorted_keys = sorted(to_embed_phylogenies.keys())
    to_embed_list_patients = [to_embed_phylogenies[key] for key in to_embed_sorted_keys]
    to_embed_data = TumorDataset(to_embed_list_patients)

    # compute the set of labels to be considered, based on the number of occurrences in the training set
    if args.min_label_occurrences > 0:
        train_data.remove_infreq_labels(args.min_label_occurrences)
        to_embed_data.replace_label_set(train_data.node_labels(), replace_with='empty')

    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=args.random_seed)
    to_embed_data.sample_one_graph_per_patient(rd_seed=args.random_seed)

    # convert the datasets into TorchTumorDataset objects, using the mapping of node labels computed from the training data also in the dataset with the phylogenies to embed
    train_torch_data = TorchTumorDataset(train_data, node_encoding_type=args.node_encoding_type)
    to_embed_torch_data = TorchTumorDataset(to_embed_data, node_encoding_type=args.node_encoding_type, known_labels_mapping=train_torch_data.node_labels_mapping)

    # create a TumorGraphGNN instance with input size based on the labels in the dataset
    model = TumorGraphGNN(
        n_node_labels=len(train_data.node_labels()),
        h_1=args.h_1,
        h_2=args.h_2,
        embedding_dim=args.embedding_dim,
        dropout_prob_1=args.dropout_prob_1,
        dropout_prob_2=args.dropout_prob_2,
        batch_normalization=args.batch_normalization,
        device=device
    )

    # load the trained model
    model.load_state_dict(torch.load(args.weights, weights_only=True))

    # create the dataloader for the dataset with the phylogenies to embed
    to_embed_dataloader = Trainer.get_dataloader(to_embed_torch_data, batch_size=args.batch_size, shuffle=False)

    # compute embeddings for the input dataset with the phylogenies to embed
    embeddings = Trainer.get_embeddings(model, to_embed_dataloader, device=device)    # by construction, the embeddings are ordered by patient id

    # create a dataframe with the embeddings
    embeddings_dics = []
    for i, patient_id in enumerate(to_embed_sorted_keys):
        curr_row = {'Patient_ID': patient_id}
        for j in range(embeddings.shape[1]):
            curr_row[f'dim_{j}'] = embeddings[i][j].item()
        embeddings_dics.append(curr_row)
    embeddings_df = pd.DataFrame(embeddings_dics)

    # save the embeddings to a .csv file
    os.makedirs(os.path.dirname(args.embeddings), exist_ok=True)
    embeddings_df.to_csv(args.embeddings, index=False)
