import os
import argparse
import json
import torch
import optuna
from optuna.trial import TrialState
from optuna.importance import get_param_importances
from functools import partial
import utils as Utils
from survival import TrainerSurvival
from survival import SurvivalGNN

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description="Tune the hyper parameters of a survival GNN-based model to predict survival time from input phylogenetic trees")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-p', '--phylogenies', type=str, required=True, help='Path to the .txt file containing the dataset with training phylogenies')
    required.add_argument('-c', '--clinical_data', type=str, required=True, help='Path to the .csv file with pre-processed clinical data')
    required.add_argument('-o', '--weights', type=str, required=True, help='Path to the file where to save the weights of the final trained model with best found hyper parameters')

    # optional arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=4, help='Max number of CPU cores for PyTorch')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')
    parser.add_argument('--optuna_db', type=str, default='sqlite:///../optuna_db/survival_model_tuning.sqlite3', help='Output file where to store the optimization study produced to optimize the hyperparameters')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials in the hyper parameters optimization search')
    parser.add_argument('--save_plot', action='store_true', help='Saves a plot reporting the training loss over epochs')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the input dataset to be considered')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation"')
    parser.add_argument('--verbose', action='store_true', help='Print training information during training')
    parser.add_argument('--save_best_params', action='store_true', help='Saves the best hyper parameters found in the optimization search')
    parser.add_argument('-v', '--validation_prop', type=float, default=0.2, help='Proportion of patients from the input dataset to be included in the validation set during the hyper parameters optimization study')

    return parser.parse_args()

if __name__ == '__main__':
    
    # parse the command line arguments
    args = parse_args()

    # set the device
    device = Utils.get_device(args.device)

    # limit the cores used by torch
    torch.set_num_threads(args.max_n_cores)
    torch.set_num_interop_threads(args.max_n_cores)

    # create all intermediate directories to the path where the database will be saved, if they do not exist
    os.makedirs(os.path.dirname(args.optuna_db[len('sqlite:///'):]), exist_ok=True)

    # define the pruner and its parameters
    top_percentile = 70.0                                               # percentile of trials that must perform better than the current trial to be pruned
    n_startup_trials = 10                                               # number of complete trials that must be performed before the pruner starts to prune
    n_warmup_steps = 10                                                 # number of epochs that must be performed before the pruner starts to prune a trial
    patience = 5                                                        # number of epochs representing the pruning patience
    base_pruner = optuna.pruners.PercentilePruner(percentile=top_percentile, n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    composed_pruner = optuna.pruners.PatientPruner(base_pruner, patience=patience)

    # create an optuna study
    study = optuna.create_study(
        storage=args.optuna_db,
        sampler=optuna.samplers.TPESampler(seed=args.random_seed),
        direction='maximize',
        study_name=f'survival_model_tuning_{args.random_seed}',
        pruner=composed_pruner
    )
    
    # optimize the objective function
    study.optimize(
        partial(
            TrainerSurvival.tuning_objective,
            phylogenies_path=args.phylogenies,
            survival_path=args.clinical_data,
            validation_proportion=args.validation_prop,
            survival_time_label=args.event_time,
            survival_event_label=args.event,
            random_seed=args.random_seed,
            device=device
        ),
        n_trials=args.n_trials,
        n_jobs=1
    )

    # report some statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("\nStudy statistics: ")
    print(f"number of finished trials: {len(study.trials)}")
    print(f"number of pruned trials: {len(pruned_trials)}")
    print(f"number of complete trials: {len(complete_trials)}")

    # extract the best trial
    best_trial = study.best_trial

    # extract and print the best configuration of hyperparameters
    best_hyperparameters = study.best_trial.params
    print('\nBest trial hyperparameters:')
    for key, value in best_hyperparameters.items():
        print(f'{key}: {value}')
    
    # print the score of the best trial
    print(f"\nScore of the best trial: {best_trial.value}")

    # compute the importance of the hyperparameters and print it
    hyperparameter_importances = get_param_importances(study)
    print("\nHyperparameter importances:")
    for key, value in hyperparameter_importances.items():
        print(f'{key}: {value}')

    # save the best hyperparameters to a file, if required
    if args.save_best_params:
        best_params_save_path = os.path.join(os.path.dirname(args.weights), 'best_hyperparameters.json')
        os.makedirs(os.path.dirname(best_params_save_path), exist_ok=True)
        with open(best_params_save_path, 'w') as file:
            json.dump(best_hyperparameters, file, indent=4)  

    # create a TorchSurvivalDataset containing the whole input data, i.e., both training and validation data
    train_data, n_labels_train = TrainerSurvival.load_train_data(
        args.phylogenies,
        args.clinical_data,
        args.event_time,
        args.event,
        args.random_seed,
        args.min_label_occurrences,
        args.node_encoding_type
    )

    # create a model instance
    model = SurvivalGNN(
        n_node_labels=n_labels_train,
        h_1_dim=best_hyperparameters['h_1'],
        h_2_dim=best_hyperparameters['h_2'],
        hidden_dim=best_hyperparameters['hidden_dim'],
        dropout_prob_1=best_hyperparameters['dropout_prob_1'],
        dropout_prob_2=best_hyperparameters['dropout_prob_2'],
        dropout_prob_3=best_hyperparameters['dropout_prob_3'],
        batch_normalization=best_hyperparameters['batch_normalization'],
        device=device
    )

    # create the directories where to save model weights and, if required, the training plot
    os.makedirs(os.path.dirname(args.weights), exist_ok=True)
    plot_save = None
    if args.save_plot:
        plot_save = os.path.join(os.path.dirname(args.weights), 'training_plot.jpg')
        os.makedirs(os.path.dirname(plot_save), exist_ok=True)

    # train the model instance on the dataloader with the found best hyperparameters
    TrainerSurvival.train(
        model,
        train_data,
        optimizer=Utils.select_optimizer(best_hyperparameters['optimizer']),
        weight_decay=best_hyperparameters['weight_decay'],
        loss_fn=TrainerSurvival.select_loss_survival(best_hyperparameters['loss_fn']),
        margin=best_hyperparameters['margin'],
        batch_size=best_hyperparameters['batch_size'],
        val_data=None,
        plot_save=plot_save,
        verbose=args.verbose,
        epochs=best_hyperparameters['epochs'],
        lr=best_hyperparameters['lr'],
        device=device,
        save_model=args.weights,
    )