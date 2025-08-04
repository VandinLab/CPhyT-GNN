import os
import argparse
from cancer_progression import Ensemble

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description='Compare the performances of different methods for the cancer progression task, across multiple random experiment repetitions.')

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--results_dir', type=str, required=True, help='Path to the directory containing all the results from cancer progression experiments on a dataset.'\
                          'The directory must contain a subdirectory "random_seed_$r" for each experiment repetition with random seed $r.'\
                          'Each subdirectory "random_seed_$r" must contain only the subdirectories for the different considered methods produced by the script "cancer_progression_experiments.py"' \
                          'and the directory "pre_processed_data" with all data used for the experiment')
                          
    # optional arguments
    parser.add_argument('-k', '--k_values', type=int, nargs='+', default=[2, 3, 4], help='List of values of clustering sizes to be used for clustering. They must be integers greater than 1')
    parser.add_argument('-m', '--methods', type=str, nargs='+', default=['Random', 'CloMu_based', 'oncotree2vec', 'RECAP'], help='List of methods to be compared. They must be from [Random, CloMu_based, RECAP, oncotree2vec].' \
                        'Remind that RECAP cannot be applied to the AML dataset. Standard CloMu is always considered')
    parser.add_argument('-p', '--plots_path', type=str, default='../plots', help='Path to the directory where to save the plots')


    return parser.parse_args()    

if __name__ == '__main__':

    # parse command line arguments
    args = parse_args()

    # name of our GNN-based method to be compared against the baseline methods provided as input
    our_method = 'GNN'

    # add our method to the list of methods to compare
    args.methods.append(our_method)

    # create the directory for the plots if it does not exist
    os.makedirs(args.plots_path, exist_ok=True)

    # percentage of test trees for which each method for every random seed and clustering size is beated
    perc_dic = Ensemble.all_percentages(args.results_dir, args.k_values, args.methods)

    # mean percentages across different random seeds for all methods and clustering sizes
    perc_same_k = Ensemble.percentages_same_k(perc_dic)

    # separator for better readability
    print(f'\n\n{"-"*100}\n')

    # compute and print mean percentages across different random seeds and clustering sizes for all methods
    Ensemble.percentages_same_method(perc_same_k)

    # plot all percentages against
    relative_plot_path = os.path.join(args.plots_path, 'relative.pdf')
    Ensemble.plot_tidy_percentages(Ensemble.make_tidy(perc_dic), our_method, relative_plot_path)

    # compute and plot absolute percentages, i.e., the percentage of test patients for which each method is the best one
    test_scores = Ensemble.compute_all_test_scores(args.results_dir, args.k_values, args.methods)
    global_percentages = Ensemble.compute_global_percentages(test_scores)
    global_plot_path = os.path.join(args.plots_path, 'global.pdf')
    Ensemble.plot_global_percentages(Ensemble.make_tidy(global_percentages), global_plot_path)