import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def custom_palette(scores_df, x_col):
    """
    Creates a custom color palette for box plots such that different box plots have diffrent luminances of blue and too light and too dark blues are avoided.

    Parameters:
    - scores_df: dataframe to be plotted containing the scores of the simulation.
    - x_col: column of the dataframe that contains the values to be plotted on the x-axis.

    Returns:
    - palette: dictionary that maps the unique values of the x_col to custom colors.
    """

    num_categories = scores_df[scores_df['Features'] == 'GNN'][x_col].nunique()
    blues = sns.color_palette("Blues", as_cmap=True)
    custom_colors = [blues(x) for x in np.linspace(0.3, 0.8, num_categories)]
    palette = dict(zip(
        sorted(scores_df[scores_df['Features'] == 'GNN'][x_col].unique()),
        custom_colors
    ))

    return palette

def plot_simulation_I(resulst_path, simulation_name_for_plotting):
    """
    Plots the results of simulation I.

    Parameters:
    - reults_path: path to the .csv file that contains the results of the simulation.
                   The plots are stored in the parent directory of the file.
    - simulation_name_for_plotting: name of the simulation to be displayed in the plot title.
    """

    # load the file with scores
    scores = pd.read_csv(resulst_path)

    # rename some columns for plotting adn convert the type to integer
    renamed_scores = scores.rename(columns={'Random Operations': 'Number of Random Changes'})
    renamed_scores['Number of Random Changes'] = renamed_scores['Number of Random Changes'].astype(int)

    # custom luminances for the blues color palette that avoids too light and too dark blues
    palette = custom_palette(renamed_scores, 'Number of Random Changes')

    # save the plots with the results across different values of number of random changes
    plt.close()
    sns.set_theme('paper')
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    sns.set_context('paper', font_scale=3)
    ax = sns.boxplot(data=renamed_scores[renamed_scores['Features'] == 'GNN'], x='Number of Random Changes', y='Rand Index', hue='Number of Random Changes', palette=palette)
    ax.set_ylim(0.5, 1.01)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_title(f'Simulation {simulation_name_for_plotting}')
    ax.set_xlabel('Number of Random Changes')
    ax.set_ylabel('Rand Index')
    ax.tick_params(axis='both')
    ax.legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(resulst_path), 'GNN_performances.pdf'))
    plt.close()

def plot_simulation_II(resulst_path, simulation_name_for_plotting):
    """
    Plots the results of simulation II.

    Parameters:
    - reults_path: path to the .csv file that contains the results of the simulation.
                   The plots are stored in the parent directory of the file.
    - simulation_name_for_plotting: name of the simulation to be displayed in the plot title.
    """

    # load the file with scores
    scores = pd.read_csv(resulst_path)

    # rename some columns for plotting and convert the type to integer
    renamed_scores = scores.rename(columns={'Base Nodes': 'Number of Nodes'})
    renamed_scores['Number of Nodes'] = renamed_scores['Number of Nodes'].astype(int)

    # custom luminances for the blues color palette that avoids too light and too dark blues
    palette = custom_palette(renamed_scores, 'Number of Nodes')

    # save the plots with the results across different values of number of nodes
    plt.close()
    sns.set_theme('paper')
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    sns.set_context('paper', font_scale=3)
    ax = sns.boxplot(data=renamed_scores[renamed_scores['Features'] == 'GNN'], x='Number of Nodes', y='Rand Index', hue='Number of Nodes', palette=palette)
    ax.set_ylim(0.5, 1.01)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_title(f'Simulation {simulation_name_for_plotting}')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Rand Index')
    ax.legend().remove()
    ax.tick_params(axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(resulst_path), 'GNN_performances.pdf'))
    plt.close()

def plot_other_simulations(results_path, simulation_name_for_plotting):
    """
    Plots the results of simulations III, IV and V.

    Parameters:
    - results_path: path to the .csv file that contains the results of the simulation.
                   The plots are stored in the parent directory of the file.
    - simulation_name_for_plotting: name of the simulation to be displayed in the plot title.
    """

    # load the file with scores
    scores = pd.read_csv(results_path)

    # rename some columns for plotting and convert the type to integer
    renamed_scores = scores.rename(columns={'Base Nodes': 'Number of Nodes'})
    renamed_scores['Number of Nodes'] = renamed_scores['Number of Nodes'].astype(int)

    # save the plots with the results across different values of number of nodes
    plt.close()
    sns.set_theme('paper')
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    sns.set_context('paper', font_scale=3)
    ax = sns.lineplot(data=renamed_scores[renamed_scores['Features'] == 'GNN'], x='Number of Nodes', y='Rand Index', marker='o')
    ax.set_ylim(0.5, 1.01)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_title(f'Simulation {simulation_name_for_plotting}')
    ax.legend().remove()
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Rand Index')
    ax.tick_params(axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(results_path), 'performances.pdf'))
    plt.close()

def parse_args():
    """
    Parses command line arguments.

    Returns:
    - parser.parse_args(): ArgumentParser object with parsed arguments.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(description="Load the results from the application of simulations and create plots showing the performances of our GNN-based model on different synthetic datasets.")

    # required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--results_dir', type=str, help='Path to the directory that contains the results of the simulations. The plots related to each simulation are stored in the "results" subfolder with the scores computed for that simulation', required=True)

    # optional arguments
    parser.add_argument('--no_sim_I', action='store_true', help='Do not consider simulation I')
    parser.add_argument('--no_sim_II', action='store_true', help='Do not consider simulation II')
    parser.add_argument('--no_sim_III', action='store_true', help='Do not consider simulation III')
    parser.add_argument('--no_sim_IV', action='store_true', help='Do not consider simulation IV')
    parser.add_argument('--no_sim_V', action='store_true', help='Do not consider simulation V')
    
    return parser.parse_args()

if __name__ == '__main__':

    # parse the command line arguments
    args = parse_args()

    # --------------------------------------------------- SIMULATION I ---------------------------------------------------

    # plot the results of simulation I, if not excluded
    if not args.no_sim_I:

        # simulation that considers binary, branching and linear trees
        plot_simulation_I(os.path.join(args.results_dir, 'I', 'I_bin_bran_lin', 'results', 'scores.csv'), 'Ia')
        
        # simualtion that considers only branching and linear trees
        plot_simulation_I(os.path.join(args.results_dir, 'I', 'I_bran_lin', 'results', 'scores.csv'), 'Ib')

    # --------------------------------------------------- SIMULATION II ---------------------------------------------------

    # plot the results of simulation II, if not excluded
    if not args.no_sim_II:

        # simulation that considers binary, branching and linear trees
        plot_simulation_II(os.path.join(args.results_dir, 'II', 'II_bin_bran_lin', 'results', 'scores.csv'), 'IIa')
        
        # simualtion that considers only branching and linear trees
        plot_simulation_II(os.path.join(args.results_dir, 'II', 'II_bran_lin', 'results', 'scores.csv'), 'IIb')

    # --------------------------------------------------- SIMULATION III ---------------------------------------------------

    # plot the results of simulation III, if not excluded
    if not args.no_sim_III:
        plot_other_simulations(os.path.join(args.results_dir, 'III', 'results', 'scores.csv'), 'III')

    # --------------------------------------------------- SIMULATION IV ---------------------------------------------------

    # plot the results of simulation IV, if not excluded
    if not args.no_sim_IV:
        plot_other_simulations(os.path.join(args.results_dir, 'IV', 'results', 'scores.csv'), 'IV')
    # --------------------------------------------------- SIMULATION V ---------------------------------------------------

    # plot the results of simulation V, if not excluded
    if not args.no_sim_V:

        # simulation that considers exclusive, ancestry and the inverse of ancestry relations
        plot_other_simulations(os.path.join(args.results_dir, 'V', 'V_exclusive_vs_ad_vs_da', 'results', 'scores.csv'), 'Va')
        
        # simulation that considers only exclusive and ancestry relations
        plot_other_simulations(os.path.join(args.results_dir, 'V', 'V_exclusive_vs_ad', 'results', 'scores.csv'), 'Vb')
        
        # simulation that considers only ancestry and its inverse relations
        plot_other_simulations(os.path.join(args.results_dir, 'V', 'V_ad_vs_da', 'results', 'scores.csv'), 'Vc')