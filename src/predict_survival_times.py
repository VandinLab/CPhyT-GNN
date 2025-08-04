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
    parser = argparse.ArgumentParser(description="Predict survival time of a set of patients given as input the phylogenetic trees that represent tumor evolution on them. It uses a previously trained supervised GNN-based model")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-t', '--train_phylogenies', type=str, required=True, help='Path to the .txt file containing the dataset with phylogenies used to train the model')
    required.add_argument('-p', '--predict_phylogenies', type=str, required=True, help='Path to the .txt file containing the dataset with the phylogenies for which to predict survival times')
    required.add_argument('-s', '--train_survival', type=str, required=True, help='Path to the .csv file containing the dataset with survival times and events used to train the model')
    required.add_argument('-w', '--weights', type=str, required=True, help='Path to the file where the weights of a trained model supervised GNN-based model are stored')
    required.add_argument('-o', '--output', type=str, required=True, help='Path to the file where to save the predicted survival times')

    # optional arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=8, help='Max number of CPU cores for PyTorch')
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the input dataset to be considered used for training')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type used for training: "clone" or "mutation"')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size used for training')
    parser.add_argument('--h_1', type=int, default=64, help='Output size of the first GCN hidden layer used for training')
    parser.add_argument('--h_2', type=int, default=64, help='Output size of the second GCN hidden layer used for training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the first linear layer used for training')
    parser.add_argument('--dropout_prob_1', type=float, default=None, help='Dropout probability before the second GCN layer used for training')
    parser.add_argument('--dropout_prob_2', type=float, default=None, help='Dropout probability before the first linear layer used for training')
    parser.add_argument('--dropout_prob_3', type=float, default=None, help='Dropout probability before the final linear layer used for training')
    parser.add_argument('--batch_normalization', action='store_true', help='Whether batch normalization was used for training')

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

    # load the training set with survival data
    train_survival = pd.read_csv(args.train_survival)

    # load the dataset of phylogenies used for training and the dataset of phylogenies for which to predict survival times
    train_phylogenies = TrainerTumorModel.load_dataset_txt(args.train_phylogenies)
    predict_phylogenies = TrainerTumorModel.load_dataset_txt(args.predict_phylogenies)

    # create a TumorDataset object for the training set that contains patients sorted by patient id so to allow for reproducibility
    train_sorted_keys = sorted(train_phylogenies.keys())
    train_list_patients = [train_phylogenies[key] for key in train_sorted_keys]
    train_data = TumorDataset(train_list_patients)

    # create a TumorDataset object for the dataset of phylogenies for which to predict survival times
    predict_sorted_keys = sorted(predict_phylogenies.keys())
    predict_list_patients = [predict_phylogenies[key] for key in predict_sorted_keys]
    predict_data = TumorDataset(predict_list_patients)

    # create a train array of tuples with survival time and survival event for each patient using the sorted keys    
    sorted_train_df = train_survival.sort_values(by='Patient_ID')
    train_survival = np.array([sorted_train_df[args.event_time].values, sorted_train_df[args.event].values])

    # create a fake array of tuples with survival time and survival event for unseen patients
    # the survival time is set to 0 and the event is set to 0 (no event)
    predict_survival = np.array([np.zeros(len(predict_list_patients)), np.zeros(len(predict_list_patients))])

    # compute the set of labels to be considered, based on the number of occurrences in the training set
    if args.min_label_occurrences > 0:
        train_data.remove_infreq_labels(args.min_label_occurrences)
        predict_data.replace_label_set(train_data.node_labels(), replace_with='empty')

    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=args.random_seed)
    predict_data.sample_one_graph_per_patient(rd_seed=args.random_seed)

    # creates the TorchSurvivalDataset objects for both datasets
    train_torch_data = TorchSurvivalDataset(train_data, train_survival, args.node_encoding_type)
    predict_torch_data = TorchSurvivalDataset(predict_data, predict_survival, node_encoding_type=args.node_encoding_type, known_labels_mapping=train_torch_data.node_labels_mapping)

    # create a SurvivalGNN instance with input size based on the labels in the dataset
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

    # load the trained model
    model.load_state_dict(torch.load(args.weights, weights_only=True))

    # create the dataloader for the phylogenies for which to predict survival times
    predict_dataloader = TrainerSurvival.get_dataloader(predict_torch_data, batch_size=args.batch_size, shuffle=False)

    # predict survival times
    predict_predictions = TrainerSurvival.predict(model, predict_dataloader, device=device).squeeze().detach().cpu().numpy()

    # save the predicted survival times
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    predictions_np = np.array([predict_sorted_keys, predict_predictions]).T
    predictions_df = pd.DataFrame(predictions_np, columns=['Patient_ID', 'Predicted_Survival_Time'])
    predictions_df.to_csv(args.output, index=False)