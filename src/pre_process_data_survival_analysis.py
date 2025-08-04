import argparse
import os
import pandas as pd
from tumor_model import TrainerTumorModel
import utils as Utils

def pre_process_data_survival_analysis(phylogenies_path, clinical_data_path, event_time, event, save_phylogenies_path, save_clinical_data_path):
    """
    Pre-processes data for survival analysis of clusters computed using the embeddings computed by our GNN-based model.

    Parameters:
    - phylogenies_path: path to the .txt file with phylogenetic trees stored in the format required by our model.
    - clinical_data_path: path to the .xlsx file with clinical data.
    - event_time: name in the sheet with clinical data of the column with survival time.
    - event: name in the sheet with clinical data of the column with binary values indicating if death occurred.
    - save_phylogenies_path: path to the file where to save the processed phylogenetic trees.
    - save_clinical_data_path: path to the file where to save the processed clinical data.
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

    # save the dataset with phylogenetic graphs and clinical data
    os.makedirs(os.path.dirname(save_phylogenies_path), exist_ok=True)
    Utils.save_dataset_txt_survival(patients_dic, save_phylogenies_path)
    os.makedirs(os.path.dirname(save_clinical_data_path), exist_ok=True)
    clinical_data.to_csv(save_clinical_data_path, index=False)

    # print some information about the dataset
    print(f'Number of patients in the processed dataset: {len(patients_dic)}')
    mutations = []
    for patient_id in patients_dic:
        for phylogeny in patients_dic[patient_id]:
            for mutation in phylogeny.get_unique_labels():
                if mutation not in mutations:
                    if mutation != 'root' and mutation != 'empty' and mutation != 'unknown':
                        mutations.append(mutation)
    print(f'Number of mutations in the processed dataset: {len(mutations)}')
    print(f'Mutations in the processed dataset: {mutations}')

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Pre-process phylogenetic trees and clinical data for survival analysis of clusters based on the embeddings computed by our GNN-based model')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-p', '--phylogenies_path', type=str, required=True, help='Path to the .txt file with phylogenetic trees stored in the format required by our model')
    required.add_argument('-c', '--clinical_data_path', type=str, required=True, help='Path to the .xlsx file with clinical data. It must have a sheet named "Clinical_Data" containing column "Patient_ID" with the ids of patients')
    required.add_argument('--save_phylogenies_path', type=str, required=True, help='Path to the file where to save the processed phylogenetic trees for survival analysis of clusters computed using the embeddings computed by our GNN-based model')
    required.add_argument('--save_clinical_data_path', type=str, required=True, help='Path to the file where to save the processed clinical data for survival analysis of clusters computed using the embeddings computed by our GNN-based model')

    # optional arguments
    parser.add_argument('--event_time', type=str, default='OS_Month', help='Name in the sheet with clinical data of the column with survival time')
    parser.add_argument('--event', type=str, default='OS_Event', help='Name in the sheet with clinical data of the column with binary values indicating if death occurred')

    return parser.parse_args()    

if __name__ == '__main__':

    # parse the command line arguments
    args = parse_args()

    # pre-process phylogenetic data and clinical data for survival analysis of clusters computed using the embeddings computed by our GNN-based model
    pre_process_data_survival_analysis(args.phylogenies_path, args.clinical_data_path, args.event_time, args.event, args.save_phylogenies_path, args.save_clinical_data_path)
