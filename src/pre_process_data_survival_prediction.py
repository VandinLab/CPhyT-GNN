import os
import argparse
import pandas as pd
from tumor_model import TrainerTumorModel
from survival import TrainerSurvival
import utils as Utils

def pre_processing_survival_prediction(phylogenies_path, clinical_data_path, save_dir_path, event, event_time, test_proportion, random_seed):
    """
    Pre-processes phylogenetic trees and clinical data for survival time prediction, splitting also data into training and test sets.

    Parameters:
    - phylogenies_path: path to the .txt file with phylogenetic trees.
    - clinical_data_path: path to the .xlsx file with clinical data.
    - save_dir_path: path to the directory where to save the pre-processed data.
    - event: name of the column in the .xlsx file with binary values indicating if death occurred.
    - event_time: name of the column in the .xlsx file with survival time.
    - test_proportion: proportion of patients in the input dataset to be included in the test set.
    - random_seed: random seed for reproducibility.
    """

    # load the dataset with phylogenetic graphs and clinical data
    patients_dic = TrainerTumorModel.load_dataset_txt(phylogenies_path)
    clinical_data = pd.read_excel(clinical_data_path, 'Clinical_Data')
    
    # extract patient ids and clinical data for the input clinical labels
    clinical_data = clinical_data[['Patient_ID', event_time, event]]

    # maximum number of phylogenetic trees that a patient can have to be considered
    max_n_graphs = 1

    # remove patients with more tumor graphs than the input threshold
    patients_dic = Utils.remove_patients_with_uncertain_phylogeny(patients_dic, max_n_graphs)

    # remove patients with unknown values or NA values
    clinical_data = Utils.remove_unknown_values(clinical_data, [event_time, event])

    # remove patients with more samples labeled differently and keep just one row per patient in the clinical data
    clinical_data = Utils.one_row_per_patient(clinical_data, event_time, event)

    # keep only patients for whom we have both phylogenies and clinical data
    patients_dic, clinical_data = Utils.intersect_ids(patients_dic, clinical_data)

    # split phylogenies and clinical data into training and test sets
    patients_dic_train, patients_dic_test, clinical_data_train, clinical_data_test = TrainerSurvival.train_val_split(
        patients_dic,
        clinical_data,
        val_proportion=test_proportion,
        rd_seed=random_seed
    )

    # create the intermediate directories in the path where to save the pre-processed data, if they do not exist
    os.makedirs(save_dir_path, exist_ok=True)

    # save training and test data
    clinical_data_train.to_csv(os.path.join(save_dir_path, 'train_clinical_data.csv'), index=False)
    clinical_data_test.to_csv(os.path.join(save_dir_path, 'test_clinical_data.csv'), index=False)
    Utils.save_dataset_txt_survival(patients_dic_train, os.path.join(save_dir_path, 'train_phylogenies.txt'))
    Utils.save_dataset_txt_survival(patients_dic_test, os.path.join(save_dir_path, 'test_phylogenies.txt'))

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Pre-process phylogenetic trees and clinical data for survival time prediction starting from phylogenetic trees')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-p', '--phylogenies_path', type=str, required=True, help='Path to the .txt file with phylogenetic trees stored in the format required by our model')
    required.add_argument('-c', '--clinical_data_path', type=str, required=True, help='Path to the .xlsx file with clinical data. It must have a sheet named "Clinical_Data" containing column "Patient_ID" with the ids of patients')
    required.add_argument('-o', '--save_dir_path', type=str, required=True, help='Path to the directory where to save the pre-processed data')

    # optional arguments
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')
    parser.add_argument('--test_proportion', type=float, default=0.2, help='Proportion of patients in the input dataset to be included in the test set. The other are included in the training set')
    parser.add_argument('-r', '--random_seed', type=int, default=27, help='Random seed for reproducibility')

    return parser.parse_args()   

if __name__ == '__main__':

    # parse command line arguments
    args = parse_args()

    # pre-process phylogenetic trees and clinical data for survival time prediction and split data into training and test sets
    pre_processing_survival_prediction(args.phylogenies_path, args.clinical_data_path, args.save_dir_path, args.event, args.event_time, args.test_proportion, args.random_seed)