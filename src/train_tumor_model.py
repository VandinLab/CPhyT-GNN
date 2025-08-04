import os
import argparse
import torch
from tumor_model import TumorDataset
from tumor_model import TorchTumorDataset
from tumor_model import TumorGraphGNN
from tumor_model import TrainerTumorModel as Trainer
import utils as Utils
from tumor_model import GraphDistances

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description="Train a tumor graph GNN model to compute unsupervised embeddings from phylogenetic trees")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--dataset', type=str, required=True, help='Path to the .txt file containing the dataset with training phylogenies')
    required.add_argument('-o', '--weights', type=str, required=True, help='Path to the file where to save the weights of the trained model')

    # optional arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=8, help='Max number of CPU cores for PyTorch')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the input dataset to be considered')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation"')
    parser.add_argument('--loss_fn', type=str, default='MAE_loss', help='Loss function to use for training: "MAE_loss" or "MSE_loss"')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use: "Adam" or "SGD"')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print training information during training')
    parser.add_argument('--save_plot', action='store_true', help='Saves a plot reporting the training loss over epochs')
    parser.add_argument('--h_1', type=int, default=64, help='Output size of the first GCN hidden layer')
    parser.add_argument('--h_2', type=int, default=64, help='Output size of the second GCN hidden layer')
    parser.add_argument('-d', '--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dropout_prob_1', type=float, default=0.3, help='Dropout probability before the second GCN layer')
    parser.add_argument('--dropout_prob_2', type=float, default=0.3, help='Dropout probability before the final linear layer')
    parser.add_argument('--batch_normalization', action='store_true', help='Use batch normalization')

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

    # load the dataset with phylogenies
    train_data = TumorDataset(args.dataset)

    # compute the set of labels to be considered, based on the number of occurrences in the training set
    if args.min_label_occurrences > 0:
        train_data.remove_infreq_labels(args.min_label_occurrences)
    
    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=args.random_seed)

    # convert the dataset into a TorchTumorDataset object
    train_torch_data = TorchTumorDataset(train_data, node_encoding_type=args.node_encoding_type)

    # compute the tensor with the distances between all pairs of graphs in the training dataset
    train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

    # create a TumorGraphGNN instance with input size based on the labels in the training set
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

    # create all intermediate folders in the paths where to save model and plots, if they do not exist
    os.makedirs(os.path.dirname(args.weights), exist_ok=True)

    # set the path where to save the training plot
    plot_save = None
    if args.save_plot:
        plot_save = os.path.join(os.path.dirname(args.weights), 'training_loss.jpg')

    # train the model on the training set with the input parameters
    Trainer.train(
        model=model,
        train_data=train_torch_data,
        train_graph_distances=train_distances,
        loss_fn=Utils.select_loss(args.loss_fn),
        batch_size=args.batch_size,
        optimizer=Utils.select_optimizer(args.optimizer),
        weight_decay=args.weight_decay,
        plot_save=plot_save,
        verbose=args.verbose,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        save_model=args.weights
    )
