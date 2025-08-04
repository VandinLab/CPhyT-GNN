import os
import argparse
import torch
import pandas as pd
import numpy as np
import utils as Utils
from tumor_model import TrainerTumorModel
from tumor_model import TumorDataset
from survival import TorchSurvivalDataset
from survival import SurvivalGNN
from survival import TrainerSurvival

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description="Train a survival GNN-based model to predict survival time from input phylogenetic trees")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-p', '--phylogenies', type=str, required=True, help='Path to the .txt file containing the dataset with training phylogenies')
    required.add_argument('-c', '--clinical_data', type=str, required=True, help='Path to the .csv file with pre-processed clinical data')
    required.add_argument('-o', '--weights', type=str, required=True, help='Path to the file where to save the weights of the trained model')

    # optional arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=8, help='Max number of CPU cores for PyTorch')
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the input dataset to be considered')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation"')
    parser.add_argument('--loss_fn', type=str, default='SquaredMarginRankingLoss', help='Loss function to use for training: "SquaredMarginRankingLoss" or "MarginRankingLoss"')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use: "Adam" or "SGD"')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', help='Print training information during training')
    parser.add_argument('--save_plot', action='store_true', help='Saves a plot reporting the training loss over epochs')
    parser.add_argument('--h_1', type=int, default=64, help='Output size of the first GCN hidden layer')
    parser.add_argument('--h_2', type=int, default=64, help='Output size of the second GCN hidden layer')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the first linear layer')
    parser.add_argument('--dropout_prob_1', type=float, default=None, help='Dropout probability before the second GCN layer')
    parser.add_argument('--dropout_prob_2', type=float, default=None, help='Dropout probability before the first linear layer')
    parser.add_argument('--dropout_prob_3', type=float, default=None, help='Dropout probability before the final linear layer')
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

    # load the datasets with phylogenies and clinical survival data
    train_phylogenies = TrainerTumorModel.load_dataset_txt(args.phylogenies)
    train_survival = pd.read_csv(args.clinical_data)

    # create a TumorDataset object that contains patients sorted by patient id so to allow for reproducibility
    sorted_keys = sorted(train_phylogenies.keys())
    list_patients = [train_phylogenies[key] for key in sorted_keys]
    train_data = TumorDataset(list_patients)

    # create an array of tuples with survival time and survival event for each patient using the sorted keys
    sorted_train_df = train_survival.sort_values(by='Patient_ID')
    train_survival = np.array([sorted_train_df[args.event_time].values, sorted_train_df[args.event].values])

    # compute the set of labels to be considered, based on the number of occurrences in the training set
    if args.min_label_occurrences > 0:
        train_data.remove_infreq_labels(args.min_label_occurrences)
    
    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=args.random_seed)

    # create a TorchSurvivalDataset object with both training phylogenies and survival data
    train_torch_data = TorchSurvivalDataset(train_data, train_survival, node_encoding_type=args.node_encoding_type)

    # create a SurvivalGNN instance with input size based on the labels in the training set
    model = SurvivalGNN(
        n_node_labels=len(train_data.node_labels()),
        h_1_dim=args.h_1,
        h_2_dim=args.h_2,
        hidden_dim=args.hidden_dim,
        dropout_prob_1=args.dropout_prob_1,
        dropout_prob_2=args.dropout_prob_2,
        dropout_prob_3=args.dropout_prob_3,
        batch_normalization=args.batch_normalization,
        device=device
    )

    # create all intermediate folders in the paths where to save model and plots, if they do not exist
    os.makedirs(os.path.dirname(args.weights), exist_ok=True)
    plot_save = None
    if args.save_plot:
        plot_save = os.path.join(os.path.dirname(args.weights), 'training_loss.jpg')

    # train and save the model on the training set with the input parameters
    TrainerSurvival.train(
        model=model,
        train_data=train_torch_data,
        batch_size=args.batch_size,
        optimizer=Utils.select_optimizer(args.optimizer),
        weight_decay=args.weight_decay,
        loss_fn=TrainerSurvival.select_loss_survival(args.loss_fn),
        plot_save=plot_save,
        verbose=args.verbose,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        save_model=args.weights
    )