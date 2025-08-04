import os
import argparse
import pandas as pd
from survival import Survival_Prediction
import utils as Utils

def load_evaluation_scores(dir_path):
    """
    Loads the evaluation scores across all experiment repetitions from the specified directory.

    Parameters:
    - dir_path: path to the directory containing the results from survival time experiments on a dataset.

    Returns:
    - eval_scores: DataFrame containing the evaluation scores across all experiment repetitions.
    """

    # initialize the dataframe that will contain the evaluation scores across all experiment repetitions
    eval_scores = pd.DataFrame()

    # iterate over the random seed directories 
    for rd_seed_dir in Utils.listdir_nohidden(dir_path):
        
        # load the evaluation scores for the current random seed and ppend them to the dataframe
        eval_scores = pd.concat([eval_scores, pd.read_csv(os.path.join(dir_path, rd_seed_dir, 'methods_evaluation.csv'))], ignore_index=True)
    
    return eval_scores

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Compare the performances of different methods for the survival time prediction task, across multiple random experiment repetitions.')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--results_dir', type=str, required=True, help='Path to the directory containing all the results from survival time experiments on a dataset.'\
                          'The directory must contain a subdirectory "random_seed_$r" for each experiment repetition with random seed $r.'\
                          'Each subdirectory "random_seed_$r" must contain a file named "methods_evaluation.csv" output from the script "survival_prediction.py" with the evaluation scores of the considered methods on the same test set')
                          
    # optional arguments
    parser.add_argument('--censored_plot_path', type=str, default='../plots/survival_prediction/breastCancer/censored_c_index.pdf', help='Path to the .pdf file where to save the plot with the censored C-Index scores for the different methods.')
    parser.add_argument('--ipcw_plot_path', type=str, default='../plots/survival_prediction/breastCancer/ipcw_c_index.pdf', help='Path to the .pdf file where to save the plot with the IPCW C-Index scores for the different methods.')

    return parser.parse_args()

if __name__ == '__main__':

    # parse the command line arguments
    args = parse_args()
    
    # load the evaluation scores across all experiment repetitions and concatenate them into a single dataframe
    eval_scores = load_evaluation_scores(args.results_dir)

    # create and save a plot that compares the performances of the different methods across all experiment repetitions
    os.makedirs(os.path.dirname(args.censored_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.ipcw_plot_path), exist_ok=True)
    Survival_Prediction.c_index_plot(eval_scores, args.censored_plot_path)
    Survival_Prediction.c_index_ipcw_plot(eval_scores, args.ipcw_plot_path)