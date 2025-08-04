import os

# set environment variable to limit the number of threads for scikit-learn
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import json
from functools import partial
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.trial import TrialState
from optuna.importance import get_param_importances
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from survival import Survival_Prediction
from survival import Survival_Features
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from tumor_model import TrainerTumorModel
from tumor_model import TumorDataset
from tumor_model import TumorGraphGNN
from tumor_model import TorchTumorDataset
from tumor_model import GraphDistances
from survival import TrainerSurvival
from survival import SurvivalGNN
from survival import TorchSurvivalDataset
import utils as Utils

def load_data(train_phylogenies_path, test_phylogenies_path, train_clinical_data_path, test_clinical_data_path, min_label_occurrences, rd_seed):
    """
    Loads training and test data with phylogenetic trees and survival time data from the input paths.

    Parameters:
    - train_phylogenies_path: path to the file with training phylogenetic trees.
    - test_phylogenies_path: path to the file with test phylogenetic trees.
    - train_clinical_data_path: path to the file with training clinical data.
    - test_clinical_data_path: path to the file with test clinical data.

    Returns:
    - train_sorted_keys: sorted list of patient ids in the training set.
    - test_sorted_keys: sorted list of patient ids in the test set.
    - train_data: TumorDataset object containing training data.
    - test_data: TumorDataset object containing test data.
    - train_clinical_data: DataFrame containing training clinical data.
    - test_clinical_data: DataFrame containing test clinical data.
    """

    # load clinical data for training and test sets
    train_clinical_data = pd.read_csv(train_clinical_data_path)
    test_clinical_data = pd.read_csv(test_clinical_data_path)

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
        test_data.replace_label_set(train_data.node_labels(), replace_with='empty')

    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=rd_seed)
    test_data.sample_one_graph_per_patient(rd_seed=rd_seed)

    return train_sorted_keys, test_sorted_keys, train_data, test_data, train_clinical_data, test_clinical_data

def prepare_data_for_SSVM(train_embeddings, test_embeddings, train_survival_data, test_survival_data, event, event_time):
    """
    Prepares the data for training and testing the Survival Support Vector Machine (SSVM) model.

    Parameters:
    - train_embeddings: DataFrame containing training embeddings.
    - test_embeddings: DataFrame containing test embeddings.
    - train_survival_data: DataFrame containing training survival data.
    - test_survival_data: DataFrame containing test survival data.
    - event: name of the column with binary values indicating if death occurred.
    - event_time: name of the column with survival time.

    Returns:
    - X_train: features for training.
    - y_train: labels of the training set.
    - X_test: features of test patients.
    - y_test: labels of the test set.
    """

    # merge embeddings and survival data into a single dataframe for each dataset, according to patient ids
    train_data = pd.merge(train_embeddings, train_survival_data, on='Patient_ID')
    test_data = pd.merge(test_embeddings, test_survival_data, on='Patient_ID')

    # split data into features and target
    X_train = train_data.drop(columns=['Patient_ID', event_time, event])
    y_train = Surv.from_dataframe(event, event_time, train_data)
    X_test = test_data.drop(columns=['Patient_ID', event_time, event])
    y_test = Surv.from_dataframe(event, event_time, test_data)

    # normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def tuning_unsupervised_GNN(train_phylogenies_path, validation_prop, min_label_occurrences, node_encoding_type, optuna_db, n_trials, hyperparam_save, jobs, random_seed, device):
    """
    Performs a hyper parameters search for our GNN model.

    Parameters:
    - train_phylogenies_path: path to the file with training phylogenetic trees.
    - validation_prop: proportion of patients from the training set to be included in the validation set during the hyper parameters optimization study.
    - min_label_occurrences: minimum number of occurrences of a mutation in the input dataset to be considered.
    - node_encoding_type: type of node encoding to be used.
    - optuna_db: path to the SQLite database for the optimization study.
    - n_trials: number of trials in the hyper parameters optimization search.
    - hyperparam_save: path to the file where to save the best hyperparameters found.
    - jobs: number of parallel jobs to run.
    - random_seed: random seed for reproducibility.
    - device: device to use for tensor operations.

    Returns:
    - best_hyperparameters: dictionary containing the best hyperparameters found.
    """

    # define the pruner and its parameters
    top_percentile = 70.0                                               # percentile of trials that must perform better than the current trial to be pruned
    n_startup_trials = 10                                               # number of complete trials that must be performed before the pruner starts to prune
    n_warmup_steps = 10                                                 # number of epochs that must be performed before the pruner starts to prune a trial
    patience = 5                                                        # number of epochs representing the pruning patience
    base_pruner = optuna.pruners.PercentilePruner(percentile=top_percentile, n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    composed_pruner = optuna.pruners.PatientPruner(base_pruner, patience=patience)

    # create an optuna study
    study = optuna.create_study(
        storage=optuna_db,
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        direction='minimize',
        study_name=f'tumor_model_tuning_{random_seed}',
        pruner=composed_pruner
    )

    # load the training and validation data and get the set of labels in the training set
    train_torch_data, val_torch_data, train_distances, val_distances, n_labels = TrainerTumorModel.load_train_val_data(
        train_phylogenies_path,
        val_proportion=validation_prop,
        rd_seed=random_seed,
        min_label_occurrences=min_label_occurrences,
        node_encoding_type=node_encoding_type,
        device=device
    )
    
    # optimize the objective function
    study.optimize(
        partial(
            TrainerTumorModel.tuning_objective,
            train_torch_data=train_torch_data,
            val_torch_data=val_torch_data,
            train_distances=train_distances,
            val_distances=val_distances,
            n_labels=n_labels,
            random_seed=random_seed,
            device=device
        ),
        n_trials=n_trials,
        n_jobs=jobs
    )

    # report some statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print(f"number of finished trials: {len(study.trials)}")
    print(f"number of pruned trials: {len(pruned_trials)}")
    print(f"number of complete trials: {len(complete_trials)}")

    # extract the best trial
    best_trial = study.best_trial

    # extract and print the best configuration of hyperparameters
    best_hyperparameters = study.best_trial.params
    print('Best trial hyperparameters:')
    for key, value in best_hyperparameters.items():
        print(f'{key}: {value}')
    
    # print the score of the best trial
    print(f"Score of the best trial: {best_trial.value}")

    # compute the importance of the hyperparameters and print it
    hyperparameter_importances = get_param_importances(study)
    print("Hyperparameter importances:")
    for key, value in hyperparameter_importances.items():
        print(f'{key}: {value}')

    # save the best hyperparameters to a file
    os.makedirs(os.path.dirname(hyperparam_save), exist_ok=True)
    with open(hyperparam_save, 'w') as file:
        json.dump(best_hyperparameters, file, indent=4)    

    return best_hyperparameters

def tuning_supervised_GNN(train_phylogenies_path, train_clinical_data_path, event, event_time, validation_prop, n_trials, optuna_db, hyperparam_save, jobs, random_seed, device):
    """
    Performs a hyper parameters search for our supervised GNN-based model to predict survival time.

    Parameters:
    - train_phylogenies_path: path to the file with training phylogenetic trees.
    - train_clinical_data_path: path to the file with training clinical data.
    - event: name of the column with binary values indicating if death occurred.
    - event_time: name of the column with survival time.
    - validation_prop: proportion of patients from the training set to be included in the validation set during the hyper parameters optimization study.
    - n_trials: number of trials in the hyper parameters optimization search.
    - optuna_db: path to the SQLite database for the optimization study.
    - hyperparam_save: path to the file where to save the best hyperparameters found.
    - jobs: number of parallel jobs to run.
    - random_seed: random seed for reproducibility.
    - device: device to use for tensor operations.
    """

    # define the pruner and its parameters
    top_percentile = 70.0                                               # percentile of trials that must perform better than the current trial to be pruned
    n_startup_trials = 10                                               # number of complete trials that must be performed before the pruner starts to prune
    n_warmup_steps = 10                                                 # number of epochs that must be performed before the pruner starts to prune a trial
    patience = 5                                                        # number of epochs representing the pruning patience
    base_pruner = optuna.pruners.PercentilePruner(percentile=top_percentile, n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    composed_pruner = optuna.pruners.PatientPruner(base_pruner, patience=patience)

    # create an optuna study
    study = optuna.create_study(
        storage=optuna_db,
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        direction='maximize',
        study_name=f'survival_model_tuning_{random_seed}',
        pruner=composed_pruner
    )
    
    # optimize the objective function
    study.optimize(
        partial(
            TrainerSurvival.tuning_objective,
            phylogenies_path=train_phylogenies_path,
            survival_path=train_clinical_data_path,
            validation_proportion=validation_prop,
            survival_time_label=event_time,
            survival_event_label=event,
            random_seed=random_seed,
            device=device
        ),
        n_trials=n_trials,
        n_jobs=jobs
    )

    # report some statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print(f"number of finished trials: {len(study.trials)}")
    print(f"number of pruned trials: {len(pruned_trials)}")
    print(f"number of complete trials: {len(complete_trials)}")

    # extract the best trial
    best_trial = study.best_trial

    # extract and print the best configuration of hyperparameters
    best_hyperparameters = study.best_trial.params
    print('Best trial hyperparameters:')
    for key, value in best_hyperparameters.items():
        print(f'{key}: {value}')
    
    # print the score of the best trial
    print(f"Score of the best trial: {best_trial.value}")

    # compute the importance of the hyperparameters and print it
    hyperparameter_importances = get_param_importances(study)
    print("Hyperparameter importances:")
    for key, value in hyperparameter_importances.items():
        print(f'{key}: {value}')

    # save the best hyperparameters to a file
    os.makedirs(os.path.dirname(hyperparam_save), exist_ok=True)
    with open(hyperparam_save, 'w') as file:
        json.dump(best_hyperparameters, file, indent=4)  

    return best_hyperparameters  

def baseline_embeddings_experiment(
        train_phylogenies_path,
        test_phylogenies_path,
        train_clinical_data,
        test_clinical_data,
        event,
        event_time,
        random_seed,
        min_label_occurrences,
        hyper_parameters_ssvm,
        cv_folds,
        max_n_cores
    ):
    """
    Trains an SSVM on baseline embeddings and evaluates the method on the test set.

    Parameters:
    - train_phylogenies_path: path to the file with training phylogenetic trees.
    - test_phylogenies_path: path to the file with test phylogenetic trees.
    - train_clinical_data: DataFrame containing training clinical data.
    - test_clinical_data: DataFrame containing test clinical data.
    - event: name of the column with binary values indicating if death occurred.
    - event_time: name of the column with survival time.
    - random_seed: random seed for reproducibility.
    - min_label_occurrences: minimum number of occurrences of a mutation in the input dataset to be considered.
    - hyper_parameters_ssvm: dictionary containing the hyperparameters for the SSVM.
    - cv_folds: number of folds for cross-validation.
    - max_n_cores: maximum number of CPU cores for PyTorch.

    Returns:
    - test_evaluation: DataFrame containing the evaluation scores on the test set.
    """

    # print information
    print('Computing baseline embeddings...')

    # load training and test phylogenetic trees and compute binary feature vectors for training and test patients
    train_embeddings, test_embeddings = Survival_Features.get_binary_feature_vectors(
        train_phylogenies_path,
        test_phylogenies_path,
        rd_seed=random_seed,
        min_label_occurrences=min_label_occurrences,
    )

    # convert data in a format suitable for applying an SSVM
    X_train, y_train, X_test, y_test = prepare_data_for_SSVM(
        train_embeddings,
        test_embeddings,
        train_clinical_data,
        test_clinical_data,
        event,
        event_time
    )

    # print information
    print('Tuning the hyper parameters of an SSVM on training baseline embeddings...')

    # train the SSVM on training embeddings
    predictor = FastSurvivalSVM()
    predictor = Survival_Prediction.train_predictor(
        predictor,
        X_train,
        y_train,
        hyper_parameters_ssvm,
        cv_folds=cv_folds,
        n_jobs=max_n_cores
    )

    # print information
    print('Training an SSVM with best found hyper parameters on training baseline embeddings...')

    # train the predictor on the training set
    predictor.fit(X_train, y_train)

    # print information
    print('Evaluating on the test set the SSVM trained on baseline embeddings...')

    # evaluate the predictor on the test set
    predictor_name = 'Baseline SSVM'
    test_evaluation = Survival_Prediction.evaluate_ssvm(
        predictor_name,
        predictor,
        X_test,
        y_train,
        y_test,
        experiment_id=random_seed,
        regression=False
    )

    return test_evaluation

def unsupervised_GNN_experiment(
        train_phylogenies_path,
        train_sorted_keys,
        test_sorted_keys,
        train_data,
        test_data,
        train_clinical_data,
        test_clinical_data,
        event,
        event_time,
        hyper_parameters_ssvm,
        output_dir,
        no_tuning,
        validation_prop,
        min_label_occurrences,
        node_encoding_type,
        n_trials,
        random_seed,
        cv_folds,
        max_n_cores,
        device,
        save_plot,
        verbose
    ):
    """
    Trains our GNN-based unsupervised model to compute embeddings, trains an SSVM on the computed embeddings, and evaluates the method on the test set.

    Parameters:
    - train_phylogenies_path: path to the file with training phylogenetic trees.
    - train_data: TumorDataset object containing training data.
    - test_data: TumorDataset object containing test data.
    - train_clinical_data: DataFrame containing training clinical data.
    - test_clinical_data: DataFrame containing test clinical data.
    - event: name of the column with binary values indicating if death occurred.
    - event_time: name of the column with survival time.
    - hyper_parameters_ssvm: dictionary containing the hyperparameters for the SSVM.
    - output_dir: path to the directory where to save the output files.
    - no_tuning: boolean indicating if hyperparameters tuning is disabled.
    - validation_prop: proportion of patients from the training set to be included in the validation set during the hyper parameters optimization study.
    - min_label_occurrences: minimum number of occurrences of a mutation in the input dataset to be considered.
    - node_encoding_type: type of node encoding to be used.
    - n_trials: number of trials in the hyper parameters optimization search.
    - random_seed: random seed for reproducibility.
    - cv_folds: number of folds for cross-validation.
    - max_n_cores: maximum number of CPU cores for PyTorch.
    - device: device to use for tensor operations.
    - save_plot: boolean indicating if the training plot should be saved.
    - verbose: boolean indicating if training information has to be printed.

    Returns:
    - test_evaluation: DataFrame containing the evaluation scores on the test set.
    """

    # initialize the hyper parameters to be used for the GNN-based model
    hyperparameters_unsupervised_GNN = {}

    # perform a grid search so to tune the hyper parameters, if not disabled
    if not no_tuning:

        # print information
        print('Tuning unsupervised GNN to compute embeddings...')

        # path where to save the SQLite database with the optimization study
        optuna_db = os.path.join('sqlite:///', output_dir, 'unsupervised_GNN_tuning.sqlite3')

        # path where to save the best hyperparameters found
        hyperparam_save = os.path.join(output_dir, 'unsupervised_GNN_best_hyperparameters.json')

        # tune the hyper parameters of the unsupervised GNN
        hyperparameters_unsupervised_GNN = tuning_unsupervised_GNN(
            train_phylogenies_path,
            validation_prop,
            min_label_occurrences,
            node_encoding_type,
            optuna_db,
            n_trials,
            hyperparam_save,
            max_n_cores,
            random_seed,
            device
        )

        # print information
        print('Training unsupervised GNN to compute embeddings with best found hyper parameters...')

    # set the hyper parameters to default values, if not tuned
    else:

        # print information
        print('Training unsupervised GNN to compute embeddings with default hyper parameters...')

        # set the hyper parameters to default values
        hyperparameters_unsupervised_GNN = {
            'h_1': 64,
            'h_2': 64,
            'embedding_dim': 32,
            'dropout_prob_1': 0.3,
            'dropout_prob_2': 0.3,
            'batch_normalization': True,
            'batch_size': 16,
            'weight_decay': 1e-4,
            'loss_fn': 'MAE_loss',
            'optimizer': 'Adam',
            'lr': 1e-3,
            'epochs': 50
        }

    # number of labels that appear in the training set
    n_labels_train = len(train_data.node_labels())

    # create a model instance
    model = TumorGraphGNN(
        n_node_labels=n_labels_train,
        h_1=hyperparameters_unsupervised_GNN['h_1'],
        h_2=hyperparameters_unsupervised_GNN['h_2'],
        embedding_dim=hyperparameters_unsupervised_GNN['embedding_dim'],
        dropout_prob_1=hyperparameters_unsupervised_GNN['dropout_prob_1'],
        dropout_prob_2=hyperparameters_unsupervised_GNN['dropout_prob_2'],
        batch_normalization=hyperparameters_unsupervised_GNN['batch_normalization'],
        device=device
    )

    # create the TorchTumorDataset for the training set
    train_torch_data = TorchTumorDataset(train_data, node_encoding_type=node_encoding_type)

    # compute the tensor with the distances between all pairs of graphs in the training dataset
    train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

    # path where to save the weights of the trained model
    unsupervised_GNN_weights = os.path.join(output_dir, 'unsupervised_GNN_weights.pth')

    # if required, save the training plot
    plot_save = None
    if save_plot:
        plot_save = os.path.join(output_dir, 'unsupervised_GNN_training_plot.jpg')

    # train the model instance on the training set
    TrainerTumorModel.train(
        model,
        train_torch_data,
        train_distances,
        loss_fn=Utils.select_loss(hyperparameters_unsupervised_GNN['loss_fn']),
        optimizer=Utils.select_optimizer(hyperparameters_unsupervised_GNN['optimizer']),
        weight_decay=hyperparameters_unsupervised_GNN['weight_decay'],
        batch_size=hyperparameters_unsupervised_GNN['batch_size'],
        val_data=None,
        val_graph_distances=None,
        plot_save=plot_save,
        verbose=verbose,
        epochs=hyperparameters_unsupervised_GNN['epochs'],
        lr=hyperparameters_unsupervised_GNN['lr'],
        early_stopping_tolerance=None,
        save_model=unsupervised_GNN_weights,
        device=device
    )

    # print information
    print('Computing unsupervised GNN-based embeddings...')

    # convert the test set into a TorchTumorDataset, using the mapping of node labels computed from the training data
    test_torch_data = TorchTumorDataset(test_data, node_encoding_type=node_encoding_type, known_labels_mapping=train_torch_data.node_labels_mapping)

    # compute the embeddings for the training and test datasets using the trained model
    train_embeddings, test_embeddings = Survival_Features.get_embeddings(model, train_torch_data, test_torch_data, train_sorted_keys, test_sorted_keys, device=device)

    # convert data in a format suitable for applying an SSVM
    X_train, y_train, X_test, y_test = prepare_data_for_SSVM(
        train_embeddings,
        test_embeddings,
        train_clinical_data,
        test_clinical_data,
        event,
        event_time
    )

    # print information
    print('Tuning the hyper parameters of an SSVM on training GNN-based embeddings...')

    # train the SSVM on training embeddings
    predictor = FastSurvivalSVM()
    predictor = Survival_Prediction.train_predictor(
        predictor,
        X_train,
        y_train,
        hyper_parameters_ssvm,
        cv_folds=cv_folds,
        n_jobs=max_n_cores
    )

    # print information
    print('Training an SSVM with best found hyper parameters on training GNN-based embeddings...')

    # train the predictor on the training set
    predictor.fit(X_train, y_train)

    # print information
    print('Evaluating on the test set the SSVM trained on GNN-based embeddings...')

    # evaluate the predictor on the test set
    predictor_name = 'GNN SSVM'
    test_evaluation = Survival_Prediction.evaluate_ssvm(
        predictor_name,
        predictor,
        X_test,
        y_train,
        y_test,
        experiment_id=random_seed,
        regression=False
    )

    return test_evaluation

def supervised_GNN_experiment(
        train_phylogenies_path,
        train_clinical_data_path,
        train_data,
        test_data,
        train_clinical_data,
        test_clinical_data,
        event,
        event_time,
        validation_prop,
        output_dir,
        no_tuning,
        n_trials,
        save_plot,
        verbose,
        max_n_cores,
        random_seed,
        device
    ):
    """
    Trains our GNN-based supervised model to predict survival times and evaluates the method on the test set.

    Parameters:
    - train_phylogenies_path: path to the file with training phylogenetic trees.
    - train_clinical_data_path: path to the file with training clinical data.
    - train_data: TumorDataset object containing training data.
    - test_data: TumorDataset object containing test data.
    - train_clinical_data: DataFrame containing training clinical data.
    - test_clinical_data: DataFrame containing test clinical data.
    - event: name of the column with binary values indicating if death occurred.
    - event_time: name of the column with survival time.
    - validation_prop: proportion of patients from the training set to be included in the validation set during the hyper parameters optimization study.
    - output_dir: path to the directory where to save the output files.
    - no_tuning: boolean indicating if hyperparameters tuning is disabled.
    - n_trials: number of trials in the hyper parameters optimization search.
    - save_plot: boolean indicating if the training plot should be saved.
    - verbose: boolean indicating if training information has to be printed.
    - max_n_cores: maximum number of CPU cores for PyTorch.
    - random_seed: random seed for reproducibility.
    - device: device to use for tensor operations.

    Returns:
    - test_evaluation_df: DataFrame containing the evaluation scores on the test set.
    """

    # initialize the hyper parameters to be used for the supervised GNN-based model
    hyperparameters_supervised_GNN = {}

    # perform a grid search so to tune the hyper parameters, if not disabled
    if not no_tuning:

        # print information
        print('Tuning supervised GNN to predict survival time...')

        # path where to save the SQLite database with the optimization study
        optuna_db = os.path.join('sqlite:///', output_dir, 'supervised_GNN_tuning.sqlite3')

        # path where to save the best hyperparameters found
        hyperparam_save = os.path.join(output_dir, 'supervised_GNN_best_hyperparameters.json')

        # tune the hyper parameters of the supervised GNN
        hyperparameters_supervised_GNN = tuning_supervised_GNN(
            train_phylogenies_path,
            train_clinical_data_path,
            event,
            event_time,
            validation_prop,
            n_trials,
            optuna_db,
            hyperparam_save,
            max_n_cores,
            random_seed,
            device
        )

        # print information
        print('Training supervised GNN to predict survival time with best found hyper parameters...')

    # set the hyper parameters to default values, if not tuned
    else:

        # print information
        print('Training supervised GNN to predict survival time with default hyper parameters...')

        # set the hyper parameters to default values
        hyperparameters_supervised_GNN = {
            'h_1': 64,
            'h_2': 64,
            'hidden_dim': 32,
            'dropout_prob_1': 0.3,
            'dropout_prob_2': 0.3,
            'dropout_prob_3': 0.3,
            'batch_normalization': True,
            'node_encoding_type': 'clone',
            'batch_size': 16,
            'weight_decay': 1e-4,
            'loss_fn': 'SquaredMarginRankingLoss',
            'margin': 1.0,
            'optimizer': 'Adam',
            'lr': 1e-3,
            'epochs': 50
        }

    # number of labels that appear in the training set
    n_labels_train = len(train_data.node_labels())

    # create a model instance
    model = SurvivalGNN(
        n_node_labels=n_labels_train,
        h_1_dim=hyperparameters_supervised_GNN['h_1'],
        h_2_dim=hyperparameters_supervised_GNN['h_2'],
        hidden_dim=hyperparameters_supervised_GNN['hidden_dim'],
        dropout_prob_1=hyperparameters_supervised_GNN['dropout_prob_1'],
        dropout_prob_2=hyperparameters_supervised_GNN['dropout_prob_2'],
        dropout_prob_3=hyperparameters_supervised_GNN['dropout_prob_3'],
        batch_normalization=hyperparameters_supervised_GNN['batch_normalization'],
        device=device
    )

    # path where to save the weights of the trained model
    supervised_GNN_weights = os.path.join(output_dir, 'supervised_GNN_weights.pth')

    # if required, save the training plot
    plot_save = None
    if save_plot:
        plot_save = os.path.join(os.path.dirname(output_dir), 'supervised_GNN_training_plot.jpg')

    # create train and test arrays of tuples with survival time and survival event for each patient using the sorted keys
    sorted_train_df = train_clinical_data.sort_values(by='Patient_ID')
    train_survival = np.array([sorted_train_df[event_time].values, sorted_train_df[event].values])
    sorted_test_df = test_clinical_data.sort_values(by='Patient_ID')
    test_survival = np.array([sorted_test_df[event_time].values, sorted_test_df[event].values])

    # create the TorchSurvivalDataset objects for training and test sets
    train_torch_data = TorchSurvivalDataset(train_data, train_survival, node_encoding_type=hyperparameters_supervised_GNN['node_encoding_type'])
    test_torch_data = TorchSurvivalDataset(test_data, test_survival, node_encoding_type=hyperparameters_supervised_GNN['node_encoding_type'], known_labels_mapping=train_torch_data.node_labels_mapping)

    # train the model instance on the dataloader with the found best hyperparameters
    TrainerSurvival.train(
        model,
        train_torch_data,
        optimizer=Utils.select_optimizer(hyperparameters_supervised_GNN['optimizer']),
        weight_decay=hyperparameters_supervised_GNN['weight_decay'],
        loss_fn=TrainerSurvival.select_loss_survival(hyperparameters_supervised_GNN['loss_fn']),
        margin=hyperparameters_supervised_GNN['margin'],
        batch_size=hyperparameters_supervised_GNN['batch_size'],
        val_data=None,
        plot_save=plot_save,
        verbose=verbose,
        epochs=hyperparameters_supervised_GNN['epochs'],
        lr=hyperparameters_supervised_GNN['lr'],
        device=device,
        save_model=supervised_GNN_weights,
    )

    # print information
    print('Evaluating on the test set the supervised GNN-based model...')

    # create the test dataloader
    test_dataloader = TrainerSurvival.get_dataloader(test_torch_data, batch_size=hyperparameters_supervised_GNN['batch_size'], shuffle=False)

    # predict test survival times
    test_predictions = TrainerSurvival.predict(model, test_dataloader, device=device).squeeze().detach().cpu().numpy()

    # convert survival times into risk scores
    test_risks = -test_predictions

    # convert the train and test survival events to boolean
    train_events = train_survival[1].astype(np.bool)
    test_events = test_survival[1].astype(np.bool)

    # compute the concordance index on the test set
    test_c_index = concordance_index_censored(test_events, test_survival[0], test_risks)

    # compute the C-Index IPCW on the test set
    train_struct_array = Surv.from_arrays(train_events, train_survival[0])
    test_struct_array = Surv.from_arrays(test_events, test_survival[0])
    test_c_index_ipcw = concordance_index_ipcw(train_struct_array, test_struct_array, test_risks)

    # create a dataframe with the evaluation results for the test set
    test_evaluation_df = pd.DataFrame(
        [{
            'Experiment ID' : random_seed,
            'Predictor': 'Supervised GNN',
            'C-Index Censored': test_c_index[0],
            'C-Index IPCW': test_c_index_ipcw[0]
        }]
    )

    return test_evaluation_df

def save_evaluation_scores(evaluation_df, experiment_id, save_path):
    """
    Saves the input evaluation dataframe to the input save_path.
    If a dataframe already exists at save_path, it is loaded and updated with the new evaluation scores.
    If entries with the same experiment_id and predictors are already present in the dataframe, then they are updated.

    Parameters:
    - evaluation_df: pandas dataframe with the evaluation metrics computed for the predictor on the input features.
                        It must contain the columns: 'Predictor', 'C-Index Censored', 'C-Index IPCW', 'Integrated Brier Score', 'Experiment ID'.
    - experiment_id: integer with the id of the experiment.
    - save_path: string with the path where to save the evaluation dataframe.

    """

    # a dataframe with evaluation scores already exists at save_path
    if os.path.exists(save_path):

        # load the dataframe and update it with the new evaluation scores
        previous_train_evaluation = pd.read_csv(save_path)
        predictors = evaluation_df['Predictor'].unique()
        for predictor in predictors:
            previous_train_evaluation = previous_train_evaluation[
                ~(
                    (previous_train_evaluation['Experiment ID'] == experiment_id) &
                    (previous_train_evaluation['Predictor'] == predictor)
                )
            ] # remove the results with the same experiment id and predictor, because we update them
        save_df = pd.concat([previous_train_evaluation, evaluation_df], ignore_index=True)
    
    # no dataframe with evaluation scores exists at save_path yet
    else:

        # the dataframe to save is the input evaluation dataframe
        save_df = evaluation_df
    
    # save the evaluation dataframe
    save_df.to_csv(save_path, index=False)

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description="Training of survival prediction models on training data and evaluation on test data")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--data_dir', type=str, required=True, help='Path to the directory containing the input data files: "train_phylogenies.txt", "test_phylogenies.txt", "train_clinical_data.csv" and "test_clinical_data.csv"')
    required.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the directory where to save the output files')

    # optional arguments
    parser.add_argument('--no_baseline', action='store_true', help='If set, the baseline Survival Support Vector Machine on binary feature vectors is not considered')
    parser.add_argument('--no_supervised_GNN', action='store_true', help='If set, the supervised GNN-based model to predict survival time is not considered')
    parser.add_argument('--no_unsupervised_GNN', action='store_true', help='If set, the Survival Support Vector Machine on unsupervised GNN-based embeddings is not considered')
    parser.add_argument('--no_tuning', action='store_true', help='If set, the hyper parameters tuning is not performed for the GNN-based models')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: "cuda", "cpu" or "mps"')
    parser.add_argument('--max_n_cores', type=int, default=4, help='Max number of CPU cores for PyTorch')
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')
    parser.add_argument('--min_label_occurrences', type=int, default=0, help='Minimum number of occurrences of a mutation in the input dataset to be considered')
    parser.add_argument('--cv_folds', type=int, default=4, help='Number of folds for cross-validation')
    parser.add_argument('--alpha_ssvm', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], help='List of regularization parameters for the SSVM to be considered in the grid search')
    parser.add_argument('--max_iter_ssvm', type=int, nargs='+', default=[50, 100, 200, 500], help='List of maximum number of iterations for the SSVM to be considered in the grid search')
    parser.add_argument('--n_trials', type=int, default=300, help='Number of trials in the hyper parameters optimization search')
    parser.add_argument('-v', '--validation_prop', type=float, default=0.2, help='Proportion of patients from the training set to be included in the validation set during the hyper parameters optimization studies')
    parser.add_argument('--save_plot', action='store_true', help='Saves the training plots for the unsupervised and supervised GNN reporting the training loss over epochs')
    parser.add_argument('--node_encoding_type', type=str, default='clone', help='Node encoding type: "clone" or "mutation"')
    parser.add_argument('--verbose', action='store_true', help='Print training information during training')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    # ------------------------------------------------------ LOAD DATA ------------------------------------------------------

    # parse command line arguments
    args = parse_args()

    # print information
    print('Loading data...')
    
    # set the device to use for tensor operations
    device = Utils.get_device(args.device)
    print(f"Using device: {device}")

    # limit the cores used by torch
    torch.set_num_threads(args.max_n_cores)
    torch.set_num_interop_threads(args.max_n_cores)

    # paths to training and test phylogenetic trees and survival time data
    train_phylogenies_path = os.path.join(args.data_dir, 'train_phylogenies.txt')
    test_phylogenies_path = os.path.join(args.data_dir, 'test_phylogenies.txt')
    train_clinical_data_path = os.path.join(args.data_dir, 'train_clinical_data.csv')
    test_clinical_data_path = os.path.join(args.data_dir, 'test_clinical_data.csv')

    # load data
    train_sorted_keys, test_sorted_keys, train_data, test_data, train_clinical_data, test_clinical_data = load_data(
        train_phylogenies_path,
        test_phylogenies_path,
        train_clinical_data_path,
        test_clinical_data_path,
        args.min_label_occurrences,
        args.random_seed
    )

    # set the hyper parameters for the grid search on the SSVM
    hyper_parameters_ssvm = {
        "alpha": args.alpha_ssvm,
        "rank_ratio": [1.0],
        "max_iter": args.max_iter_ssvm
    }

    # creathe the output directory, if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the dataframe that will contain the evaluation scores on the test set for all methods
    test_evaluation = pd.DataFrame()

    # ------------------------------------------------------ SSVM ON BASELINE FEATURES ------------------------------------------------------

    # consider the baseline model, if not excluded
    if not args.no_baseline:

        # train an SSVM on baseline embeddings and evaluate the method on the test set
        test_evaluation_baseline = baseline_embeddings_experiment(
            train_phylogenies_path,
            test_phylogenies_path,
            train_clinical_data,
            test_clinical_data,
            args.event,
            args.event_time,
            args.random_seed,
            args.min_label_occurrences,
            hyper_parameters_ssvm,
            args.cv_folds,
            args.max_n_cores
        )

        # concatenate the evaluation scores of the baseline model to the dataframe with the evaluation scores of all methods
        test_evaluation = pd.concat([test_evaluation, test_evaluation_baseline], ignore_index=True)

    # ------------------------------------------------------ SSVM ON UNSUPERVISED GNN-BASED FEATURES ------------------------------------------------------

    # consider the unsupervised GNN-based model, if not excluded
    if not args.no_unsupervised_GNN:
        
        # train the unsupervised GNN-based model to compute embeddings, train an SSVM on the computed embeddings and evaluate the method on the test set
        test_evaluation_GNN = unsupervised_GNN_experiment(
            train_phylogenies_path,
            train_sorted_keys,
            test_sorted_keys,
            train_data,
            test_data,
            train_clinical_data,
            test_clinical_data,
            args.event,
            args.event_time,
            hyper_parameters_ssvm,
            args.output_dir,
            args.no_tuning,
            args.validation_prop,
            args.min_label_occurrences,
            args.node_encoding_type,
            args.n_trials,
            args.random_seed,
            args.cv_folds,
            args.max_n_cores,
            device,
            args.save_plot,
            args.verbose
        )

        # concatenate the evaluation scores of the GNN-based model to the dataframe with the evaluation scores of all methods
        test_evaluation = pd.concat([test_evaluation, test_evaluation_GNN], ignore_index=True)

    # ------------------------------------------------------ SUPERVISED GNN-BASED SURVIVAL TIME MODEL ------------------------------------------------------

    # consider the supervised GNN-based model, if not excluded
    if not args.no_supervised_GNN:

        # train the supervised GNN-based model to predict survival time and evaluate the method on the test set
        test_evaluation_supervised_GNN = supervised_GNN_experiment(
            train_phylogenies_path,
            train_clinical_data_path,
            train_data,
            test_data,
            train_clinical_data,
            test_clinical_data,
            args.event,
            args.event_time,
            args.validation_prop,
            args.output_dir,
            args.no_tuning,
            args.n_trials,
            args.save_plot,
            args.verbose,
            args.max_n_cores,
            args.random_seed,
            device
        )

        # concatenate the evaluation scores of the supervised GNN-based model to the dataframe with the evaluation scores of all methods
        test_evaluation = pd.concat([test_evaluation, test_evaluation_supervised_GNN], ignore_index=True)
    
    # ------------------------------------------------------ SAVE EVALUATION SCORES OF ALL METHODS ------------------------------------------------------
    
    # save the evaluation dataframe
    save_evaluation_scores(test_evaluation, args.random_seed, os.path.join(args.output_dir, 'methods_evaluation.csv'))