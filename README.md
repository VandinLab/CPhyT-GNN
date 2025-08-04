# System Requirements

The software is entirely written in Python, version `3.9.6`.
To run it, the installation of some non-standard packages is required.
The file requirements.txt contains all python packages required along with their version.
It is possible to automatically install all packages in the requirements.txt file so to get ready to use this software by just run on terminal:
```
pip install -r requirements.txt
```

# Data

The model we propose works on input dataset of phylogenetic trees, organized into patients.
The dataset must be saved in a `.txt` file, with an appropriate format.
To see the format required by our model, take as examples the `breast_cancer.txt` and `AML.txt` files we provide inside the `data` directory.

# Train the Model

## Custom Hyper Parameters

To train our model on an input dataset of phylogenetic trees with the objective of computing unsupervised embeddings, run the script `train_tumor_model.py` inside the `src` directory.
For usage instructions, run `python3 train_tumor_model.py -h`.
For instance, to train the model on the breast cancer dataset used in the experiments with standard optional parameters, run:
```
cd src &&
python3 train_tumor_model.py -i ../data/breast_cancer.txt -o ../weights/weights.pth
```

## Hyper Parameters Tuning

We also provide a script to perform a hyper parameter search so to find the best hyper parameters for the model, given an input dataset that is automatically split into training and validation set.
After the search, the best set of hyper parameters found is used to train a new model instance on the whole dataset, previously split into training and validation set, saving the weights afterwards.
The script is `train_tumor_model.py`, inside the `src` directory.
For usage instructions, run `python3 tune_tumor_model.py -h`.
As an example, to perform a hyper parameter search for the model on the breast cancer dataset used in the experiments, run:
```
cd src &&
python3 tune_tumor_model.py -i ../data/breast_cancer.txt -o ../weights/opt/weights.pth --save_best_params
```

# Compute Embeddings

Once a set of weights for our model is learnt, it is possible to apply the model to any dataset of phylogenetic trees representing tumor evolution for the same cancer type of those used for training.
Therefore, the model can be applied even to unseen test data.
To compute the embeddings of a dataset of phylogenetic trees, run the script `compute_embeddings.py` in the `src` directory.
For usage instructions, run `python3 compute_embeddings.py -h`.
For instance, given a training set `data/train_set.txt` used for training our model, producing weights `weights/weights.pth`, it is possible to compute the embeddings of a dataset of unseen phylogenetic trees `data/unseen_phylogenies.txt` and store the embeddings in tabular form in a file `embeddings/embeddings.csv` with the following command:
```
cd src &&
python3 compute_embeddings.py --train_set ../data/train_set.txt -i ../data/unseen_phylogenies.txt -w ../weights/weights.pth -o ../embeddings/embeddings.csv 
```

Clearly, it is also possible to compute the embeddings for the phylogenetic trees used for training, for instace with the command:
```
cd src &&
python3 compute_embeddings.py --train_set ../data/train_set.txt -i ../data/train_set.txt -w ../weights/weights.pth -o ../embeddings/embeddings.csv 
```

# Simulations

The script with the code to reproduce all the simulations we performed is `simulations.py` in the `src` folder.
It is possible to run only some of the simulations and set custom parameters, as explained running the command `python3 simulations.py -h`.
For instance, to store the results computing by running the simulations in the folder `results/simulations`, run:
```
cd src &&
python3 simulations.py -o ../results/simulations
```

Once that the script has finished and the results are correctly saved to the output folder, it is possible to create some plots to visualize them using the following command, supposing that the results are saved in `results/simulations`:
```
cd src &&
python3 plot_simulations_results.py -i ../results/simulations
```
For information on how to run the script, run `python3 plot_simulations_results.py -h`.

# Cancer Progression Modeling

In this section, we provide all details to reproduce the experiments reported in the paper for the cancer progression modeling experiments.
In particular, the steps to be followed can be summarized as:
1. pre-process data and split it into training and test set;
2. run the Ensemble Learning strategy that we propose with our model and some baselines;
3. compare the performances of all approaches.

In our paper, we ran `10` times steps `1.` and `2.`, each time with a different random seed. More specifically, the random seeds we used are: `27, 8, 14, 4, 7, 15, 25, 3, 17, 10`.
Afterwards, we compared the performances across all random seeds.

## Pre-Process Data

Before running any method, it is necessary to pre-process the input dataset so to make it compliant with CloMu requirements, convert it into the different formats required by different methods such as RECAP and oncotree2vec and split data into training and test sets.
The described operations can be performed by running the script `pre_process_data_cancer_progression.py`.
For usage instructions, run `python3 pre_process_data_cancer_progression.py -h`, inside the `src` folder.
For instance to pre-process, convert and split the `breastCancer.npy` dataset, run:
```
cd src &&
python3 pre_process_data_cancer_progression.py -d ../data/cancer_progression/breastCancer.npy
```
Remind that in pur experiments we added the `--gene_level_analysis` flag for the AML dataset, but not for the breast cancer dataset.
Therefore, to pre-process, convert and split the `AML.npy` dataset, run:
```
cd src &&
python3 pre_process_data_cancer_progression.py -i ../data/cancer_progression/AML.npy --gene_level_analysis --max_tree_length 10
```

## Run Cancer Progression Experiments

Once that the pre-processing script has been run on some dataset, it is possible to run the script `cancer_progression_experiments.py`, reproducing the experiments related to cancer progression modeling explained in the paper.

The script automatically runs all methods considered in the paper, but there are option to exclude every method, so to be able to run the script with only the desired methods (see `python3 cancer_progression_experiments.py -h` for usage).
In particular, notice that to run the random method it is necessary that our GNN-based method has already been run, because the former creates clusters of the same size of our method.

To run oncotree2vec and RECAP, they must be cloned from their original github repositories, properly configured following the related instructions and applied to the converted versions of the training set produced by the previously applied script `pre_process_data_cancer_progression.py`.
If oncotree2vec or RECAP have not been applied to the training set yet, then it is not possible to include them when running `cancer_progression_experiments.py`, so exclude them with the options provided by the script.
Remind also that RECAP cannot be applied to the AML dataset due to the violation of the infinite sites assumption.

The scripts automatically performs an intensive hyper parameter search for our GNN-based model before training and this will require a considerable amount of time.
Therefore, we added the option `--no_tuning` so to avoid hyper parameters tuning and directly train the GNN-based model with standard parameters.

As an example, given all the required data for the breast cancer dataset inside the folder `./results/cancer_progression_modeling/breastCancer/random_seed_27/pre_processed_data`, it is possible to run the experiment with:
```
cd src &&
python3 cancer_progression_experiments.py -i ../results/cancer_progression_modeling/breastCancer/random_seed_27/pre_processed_data -d 64 --max_tree_length 9 --infinite_sites --no_tuning
```
For the AML dataset with all required data in the folder `./results/cancer_progression_modeling/AML/random_seed_27/pre_processed_data`:
```
cd src &&
python3 cancer_progression_experiments.py -i ../results/cancer_progression_modeling/AML/random_seed_27/pre_processed_data -d 32 --gamma 2 --max_tree_length 10 --no_tuning --no_RECAP
```

## Compare Methods

The script `compare_methods_cancer_progression.py` can be used to reproduce the plots reported in the paper for cancer progression modeling and provides more detailed information about the performances of the different methods considered for cancer progression modeling.
The script must be run on the results computed and saved by `cancer_progression_experiments.py`.
See `python3 compare_methods_cancer_progression.py -h` for usage.

For instance, after having computed the results from all methods, to obtain evaluation results and plots for the breast cancer dataset, run:
```
cd src &&
python3 compare_methods_cancer_progression.py -i ../results/cancer_progression_modeling/breastCancer -p ../plots/cancer_progression/breastCancer
```

For the AML dataset, run:
```
cd src &&
python3 compare_methods_cancer_progression.py -i ../results/cancer_progression_modeling/AML -m Random CloMu_based oncotree2vec -p ../plots/cancer_progression/AML
```

# Survival

## Survival Analysis

In what follows, we describe the steps to reproduce the experiments reported in the paper regarding survival analysis of the clusters computed on the embeddings extracted by our GNN-based method.

First, it is necessary to run the script `pre_process_data_survival_analysis.py` (see `python3 pre_process_data_survival_analysis.py -h` for usage information) to pre-process and merge clinical survival data with phylogenetic data.
For instance, to pre-process the breast cancer data we provide in the `data/survival` directory, run:
```
cd src &&
python3 pre_process_data_survival_analysis.py -p ../data/survival/breast_phylogenies.txt -c ../data/survival/breast_clinical_data.xlsx --save_phylogenies_path ../data/survival/pre_processed/breast_phylogenies.txt --save_clinical_data_path ../data/survival/pre_processed/breast_clinical_data.csv
```

Second, a GNN-based model has to be trained on the pre-processed dataset of phylogenetic trees, so to make it learn unsupervised embeddings for patients.
This can be done using either the script `train_tumor_model.py` or the script `tune_tumor_model.py`, as explained in the first sections of this guide.
For instance:
```
cd src &&
python3 train_tumor_model.py -i ../data/survival/pre_processed/breast_phylogenies.txt -o ../results/survival/survival_analysis/breastCancer/weights.pth
```

Now that the weights for our GNN-based model have been optimized based on the dataset of phylogenetic trees, we use them to compute embeddings for the phylogenetic trees, we cluster them and we analyze the survival distribution of each cluster. To do this, we provide the script `survival_analysis.py`.
For example, following the previously run command, run:
```
cd src &&
python3 survival_analysis.py -p ../data/survival/pre_processed/breast_phylogenies.txt -c ../data/survival/pre_processed/breast_clinical_data.csv -w ../results/survival/survival_analysis/breastCancer/weights.pth
```
See `python3 survival_analysis.py -h` for a more detailed usage description.

## Survival Time Prediction

### Supervised GNN-Based Survival Model

As explained in the paper, along with our unsupervised GNN-based model to compute embeddings and then use them to train a Survival Support Vector Machine, we also propose an end-to-end GNN-based model to predict survival time.
The model can be trained using the script `train_survival_model.py` and its hyper parameters can be tuned using `tune_survival_model.py`, similarly to our classical GNN-based model.
As usual, refer to `python3 train_survival_model.py -h` and `python3 tune_survival_model.py -h` for usages information.
For instance, given a training set of phylogenetic trees with relative path `../data/survival/pre_processed/train_phylogenies.txt` from the `src` directory and the corresponding survival data in `../data/survival/pre_processed/train_clinical_data.csv`, it is possible to train a supervised GNN-based survival model and save its weights with the command:
```
python3 train_survival_model.py -p ../data/survival/pre_processed/train_phylogenies.txt -c ../data/survival/pre_processed/train_clinical_data.csv -o ../weights/weights.pth
```

Given a trained model, it is possible to use it to predict survival times for an input dataset of phylogenetic trees using `predict_survival_times.py` (see `python3 predict_survival_times.py -h` for detailed usage).
For instance, run:
```
python3 predict_survival_times.py -t ../data/survival/pre_processed/train_phylogenies.txt -p ../data/survival/pre_processed/unseen_phylogenies.txt -s ../data/survival/pre_processed/train_clinical_data.csv -w ../weights_surv/weights.pth -o ../results/survival/survival_times.csv
```
where the paths to the files provided as input arguments in the example must refer to previously generated data.
Notice that the predicted survival times are not normalized to be in the range `[0, +inf)` and so do not represent survival times, rather scores proportional to survival times.
Indeed, as commonly used in the literature, our model treates the survival time prediction task as a ranking task rather than a regression problem.
Therefore, it is not that important that the predicted survival times exactly match the real ones, but we wish that the ranking is analogous.

### Experiments

To reproduce the experiments that we performed for the survival time prediction task, it is necessary to follow the following steps:
1. pre-process phylogenetic trees and clinical data, merge them based on patient ids and split data into traning and test sets;
2. run an experiment in which supervised GNN model, unsupervised GNN model and baseline are trained and applied to predict survival times on test data;
3. compare the performances of the three methods.

The reults reported in the paper are obtained repeating steps `1.` and `2.` with the following `10` random seeds: `27, 8, 14, 4, 7, 15, 25, 3, 17, 10`.

#### Pre-Process Data

To pre-process data and split it into training and test sets, it is possible to use the script `pre_process_data_survival_prediction.py` (see `python3 pre_process_data_survival_prediction.py -h` for usage).
For instance, run:
```
cd src &&
python3 pre_process_data_survival_prediction.py -p ../data/survival/breast_phylogenies.txt -c ../data/survival/breast_clinical_data.xlsx -o ../results/survival/survival_prediction/breastCancer/random_seed_27/pre_processed_data -r 27
```
to pre-process the breast cancer data for survival that we provide in the `data` directory.

#### Run Survival Time Prediction Experiments

The script `survival_prediction.py` trains and evaluates the three methods we consider for survival time prediction: Survival SVM on baseline features, Survival SVM on unsupervised GNN-based features and supervised GNN-based model.
The three methods are trained on the same training set and evaluated on the same test set.
The script has several option that can be enabled, including the exclusion of the desired method, making the experiment run only on a subset of them.
As usual, refer to `python3 survival_prediction.py -h` for further information.

As an example, all the three methods can be trained on a training set and evaluated on a test set stored in the directory `results/survival/survival_prediction/breastCancer/random_seed_27/pre_processed_data` using the command:
```
cd src &&
python3 survival_prediction.py -i ../results/survival/survival_prediction/breastCancer/random_seed_27/pre_processed_data -o ../results/survival/survival_prediction/breastCancer/random_seed_27 -r 27
```

#### Compare Survival Methods

When some repetitions of the survival time prediction experiment have produced the results, it is possible to compare the considered methods among the three proposed across all experiment repetitions using the script `compare_methods_survival.py` (`python3 compare_methods_survival.py -h` for usage information).
The script gathers all `methods_evaluation.csv` files produces by different experiment runs and produces a faceted plot comparing the considered survival methods.

For instance, if the results are stored in the directory `results/survival/survival_prediction/breastCancer`, then it is possible to run:
```
python3 compare_methods_survival.py -i ../results/survival/survival_prediction/breastCancer
```