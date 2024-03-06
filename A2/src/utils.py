from __future__ import print_function

# import core packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose
import pickle

# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# sklearn models
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# parallel computing packages
from itertools import product
from joblib import Parallel, delayed

import warnings
import time
warnings.filterwarnings("ignore")

# CONFIG
SEED_NUM = 42
FOURPEAK_FOLDER_NAME = 'src/section1/fourpeak/'
FLIPFLOP_FOLDER_NAME = 'src/section1/flipflop/'
KNAPSACK_FOLDER_NAME = 'src/section1/knapsack/'

# Helper functions
def CalcTime(function, *args, **kwargs):
    start_time = time.time()
    output = function(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time, output

def generate_parameter_list(parameter_dict):
    combinations = product(*parameter_dict.values())
    parameters_list = [
        dict(zip(parameter_dict.keys(), combination)) for combination in combinations
    ]

    return parameters_list


"""
OPTIMISATION PROBLEMS:
1. FourPeaks
2. FlipFlop
3. KnapSack
"""
def problem_fourpeaks(t_pct):
    return mlrose.FourPeaks(t_pct=t_pct)

def problem_flipflop():
    return mlrose.FlipFlop()

def problem_knapsack(weights, values, max_weight_pct):
    # default - weights = [10, 5, 2, 8, 15]
    # values = [1, 2, 3, 4, 5]
    # max_weight_pct = 0.6
    return mlrose.Knapsack(weights=weights, 
                           values=values,
                           max_weight_pct=max_weight_pct)
"""
SEARCH ALGORITHMS:
1. RHC
2. GA
3. SA
4. MIMIC
"""
# define search algorithms
def RHC(problem, max_attempts, max_iters, restarts, random_state):
    # default - max_attempts: int = 10, max_iters: float = np.inf,
    return mlrose.random_hill_climb(problem, max_attempts=max_attempts, 
                             max_iters=max_iters, restarts=restarts, 
                             init_state=None, curve=True, 
                             random_state=random_state)

def GA(problem, pop_size, mutate_prob, max_attempt, max_iters, random_state):
    # default - pop_size = 200, mutate_prob=0.1, max_attempt=10
    return mlrose.genetic_alg(problem, pop_size=pop_size, 
                              mutation_prob=mutate_prob, max_attempts=max_attempt, 
                              max_iters=max_iters, curve=True, 
                              random_state=random_state)

def SA(problem, schedule, max_attempts, max_iters, random_state):
    # default - schedule: GeomDecay = GeomDecay(), max_attempts: int = 10,
    return mlrose.simulated_annealing(problem, schedule = schedule,
                                      max_attempts=max_attempts, max_iters=max_iters,
                                      init_state=None, curve=True,
                                      random_state=random_state)

def MIMIC(problem, pop_size, keep_pct, max_attempts, max_iters, random_state):
    # default - keep_pct: float = 0.2, max_attempts: int = 10
    return mlrose.mimic(problem, pop_size = pop_size, max_iters = max_iters,
                        keep_pct = keep_pct, max_attempts=max_attempts, curve=True,
                        random_state = random_state)

"""
Experiment functions:
1. RHC
2. GA
3. SA
4. MIMIC
"""
def run_rhc_experiment(length, fitness, experiment_params, verbose=True):

    print(f'RHC Experiment for length: {length}') if verbose else next
    # initialise parameters
    exp_list = generate_parameter_list(experiment_params)
    best_fitness = 0
    best_state = None
    best_fitness_curve = None
    best_params = {}

    # experiments
    for exp in exp_list:
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness,
            maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(fast_mode=True)
        algo = exp['algo']

        max_attempts_param = exp['max_attempts_param']
        max_iters_param = exp['max_iters_param']
        restarts_param = exp['restarts_param']

        elapsed_time, output = CalcTime(algo, problem, max_attempts_param, max_iters_param, restarts_param, SEED_NUM)
        state_rhc, fitness_rhc, fitness_curve_rhc = output
        if fitness_rhc > best_fitness:
            best_fitness = fitness_rhc
            best_state = state_rhc
            best_fitness_curve = fitness_curve_rhc

            best_params['algo'] = 'RHC'
            best_params['best_max_attempts'] = max_attempts_param
            best_params['best_max_iters'] = max_iters_param
            best_params['best_wall_clock_time'] = elapsed_time
            best_params['best_state'] = best_state
            best_params['best_fitness_curve'] = best_fitness_curve
            best_params['best_fitness_score'] = best_fitness
    
    return best_params


def run_sa_experiment(length, fitness, experiment_params, verbose=True):

    print(f'SA Experiment for length: {length}') if verbose else next
    # initialise parameters
    exp_list = generate_parameter_list(experiment_params)
    best_fitness = 0
    best_state = None
    best_fitness_curve = None
    best_params = {}

    # experiments
    for exp in exp_list:
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness,
            maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(fast_mode=True)
        
        algo = exp['algo']
        max_attempts_param = exp['max_attempts_param']
        max_iters_param = exp['max_iters_param']
        init_temp_param = exp['init_temp_param']
        decay_param = exp['decay_param']
        min_temp_param = exp['min_temp_param']
        schedule = mlrose.GeomDecay(init_temp=init_temp_param, decay=decay_param, min_temp=min_temp_param)

        elapsed_time, output = CalcTime(algo, problem, schedule, max_attempts_param, max_iters_param, SEED_NUM)
        state, fitness_score, fitness_curve = output
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_state = state
            best_fitness_curve = fitness_curve

            best_params['algo'] = 'SA'
            best_params['best_max_attempts'] = max_attempts_param
            best_params['best_max_iters'] = max_iters_param
            best_params['best_init_temp'] = init_temp_param
            best_params['best_decay'] = decay_param
            best_params['best_min_temp'] = min_temp_param
            best_params['best_wall_clock_time'] = elapsed_time
            best_params['best_state'] = best_state
            best_params['best_fitness_curve'] = best_fitness_curve
            best_params['best_fitness_score'] = best_fitness
    
    return best_params


def run_ga_experiment(length, fitness, experiment_params, verbose=True):

    print(f'GA Experiment for length: {length}') if verbose else next
    # initialise parameters
    exp_list = generate_parameter_list(experiment_params)
    best_fitness = 0
    best_state = None
    best_fitness_curve = None
    best_params = {}

    # experiments
    for exp in exp_list:
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness,
            maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(fast_mode=True)
        
        algo = exp['algo']
        max_attempts_param = exp['max_attempts_param']
        max_iters_param = exp['max_iters_param']
        pop_size_param = exp['pop_size_param']
        mutation_prob_param = exp['mutation_prob_param']

        elapsed_time, output = CalcTime(algo, problem, pop_size_param, mutation_prob_param, max_attempts_param, max_iters_param, SEED_NUM)
        state, fitness_score, fitness_curve = output
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_state = state
            best_fitness_curve = fitness_curve

            best_params['algo'] = 'GA'
            best_params['best_max_attempts'] = max_attempts_param
            best_params['best_max_iters'] = max_iters_param

            best_params['best_pop_size'] = pop_size_param
            best_params['best_mutation_prob'] = mutation_prob_param

            best_params['best_wall_clock_time'] = elapsed_time
            best_params['best_state'] = best_state
            best_params['best_fitness_curve'] = best_fitness_curve
            best_params['best_fitness_score'] = best_fitness
    
    return best_params

def run_mimic_experiment(length, fitness, experiment_params, verbose=True):

    print(f'MIMIC Experiment for length: {length}') if verbose else next
    # initialise parameters
    exp_list = generate_parameter_list(experiment_params)
    best_fitness = 0
    best_state = None
    best_fitness_curve = None
    best_params = {}

    # experiments
    for exp in exp_list:
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness,
            maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(fast_mode=True)
        
        algo = exp['algo']
        max_attempts_param = exp['max_attempts_param']
        max_iters_param = exp['max_iters_param']
        pop_size_param = exp['pop_size_param']
        keep_pct_param = exp['keep_pct_param']

        elapsed_time, output = CalcTime(algo, problem, pop_size_param, keep_pct_param, max_attempts_param, max_iters_param, SEED_NUM)
        state, fitness_score, fitness_curve = output
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_state = state
            best_fitness_curve = fitness_curve

            best_params['algo'] = 'MIMIC'
            best_params['best_max_attempts'] = max_attempts_param
            best_params['best_max_iters'] = max_iters_param

            best_params['best_pop_size'] = pop_size_param
            best_params['keep_pct'] = keep_pct_param

            best_params['best_wall_clock_time'] = elapsed_time
            best_params['best_state'] = best_state
            best_params['best_fitness_curve'] = best_fitness_curve
            best_params['best_fitness_score'] = best_fitness
    
    return best_params

"""
HYPERPARAMETER TUNING FUNCTIONS
"""
def tune_models(fitness, problem_size_list, parameters, output_path):
    result_dict = {}
    for length in problem_size_list:
        if parameters['algo'] == RHC:
            best_params = run_rhc_experiment(length, fitness, parameters)
        elif parameters['algo'] == SA:
            best_params = run_sa_experiment(length, fitness, parameters)
        elif parameters['algo'] == GA:
            best_params = run_ga_experiment(length, fitness, parameters)
        elif parameters['algo'] == MIMIC:
            best_params = run_mimic_experiment(length, fitness, parameters)
        else:
            print('Invalid algorithm found, please define algo in parameters')
        result_dict[length] = best_params
    
    with open(output_path, 'wb') as f:
        pickle.dump(result_dict, f)

def get_hyperparameters(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def put_pickle_file(filepath, file):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)


"""
PLOTTING FUNCTIONS
"""
def run_problem_size_experiment(fitness):
    length_list = list(range(10, 100, 10))
    rhc_params = {"algo": [RHC],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    sa_params = {"algo": [SA],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    ga_params = {"algo": [GA],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    mimic_params = {"algo": [MIMIC],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict = dict(), dict(), dict(), dict()    
    
    for length in length_list:
        best_params = run_rhc_experiment(length, fitness, rhc_params)
        rhc_score_dict[length] = best_params
        best_params = run_sa_experiment(length, fitness, sa_params)
        sa_score_dict[length] = best_params
        best_params = run_ga_experiment(length, fitness, ga_params)
        ga_score_dict[length] = best_params
        best_params = run_mimic_experiment(length, fitness, mimic_params)
        mimic_score_dict[length] = best_params
    
    return rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict


def run_problem_size_experiment_parallel(fitness):
    length_list = list(range(10, 100, 10))
    rhc_params = {"algo": [RHC],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    sa_params = {"algo": [SA],
                        "max_attempts_param": [100],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.5],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    ga_params = {"algo": [GA],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    mimic_params = {"algo": [MIMIC],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}
    
    rhc_score_dict = Parallel(n_jobs=-1)(delayed(run_rhc_experiment)(length, fitness, rhc_params) for length in length_list)
    sa_score_dict = Parallel(n_jobs=-1)(delayed(run_sa_experiment)(length, fitness, sa_params) for length in length_list)
    ga_score_dict = Parallel(n_jobs=-1)(delayed(run_ga_experiment)(length, fitness, ga_params) for length in length_list)
    mimic_score_dict = Parallel(n_jobs=-1)(delayed(run_mimic_experiment)(length, fitness, mimic_params) for length in length_list)
    
    return rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict

def run_problem_size_experiment_parallel_knapsack(fitness):
    length_list = list(range(10, 100, 10))
    rhc_params = {"algo": [RHC],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    sa_params = {"algo": [SA],
                        "max_attempts_param": [100],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.5],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    ga_params = {"algo": [GA],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}

    mimic_params = {"algo": [MIMIC],
                        "max_attempts_param": [1000],
                        "max_iters_param": [1000],
                        "restarts_param": [0],
                        "init_temp_param": [5],
                        "decay_param": [0.1],
                        "min_temp_param": [0.001],
                        "pop_size_param": [200],
                        "mutation_prob_param": [0.1],
                        "keep_pct_param": [0.2]}
    
    rhc_score_dict = {}
    sa_score_dict = {}
    ga_score_dict = {}
    mimic_score_dict = {}
    
    rhc_score_dict = Parallel(n_jobs=-1)(delayed(run_rhc_experiment)(length, fitness(length), rhc_params) for length in length_list)
    sa_score_dict = Parallel(n_jobs=-1)(delayed(run_sa_experiment)(length, fitness(length), sa_params) for length in length_list)
    ga_score_dict = Parallel(n_jobs=-1)(delayed(run_ga_experiment)(length, fitness(length), ga_params) for length in length_list)
    mimic_score_dict = Parallel(n_jobs=-1)(delayed(run_mimic_experiment)(length, fitness(length), mimic_params) for length in length_list)

    return rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict


# plot_size_variable('best_fitness_score', length=list(range(10, 100, 10)), rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict, 
#                    title = 'Fitness score by problem size (Fourpeaks)', y_lab = 'Fitness score', 
#                    output_path=FOURPEAK_FOLDER_NAME + 'problem_size_fitness_fourpeaks.png')

def plot_size_variable(variable, length, rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict, title, y_lab, output_path):
    lengths = np.array(length)
    rhc_score_array = np.array([rhc_score_dict[key][variable] for key in rhc_score_dict.keys()])
    sa_score_array = np.array([sa_score_dict[key][variable] for key in sa_score_dict.keys()])
    ga_score_array = np.array([ga_score_dict[key][variable] for key in ga_score_dict.keys()])
    mimic_score_array = np.array([mimic_score_dict[key][variable] for key in mimic_score_dict.keys()])

    plt.figure(figsize=(10, 6))

    plt.plot(lengths, rhc_score_array, label='RHC ', marker='o')
    plt.plot(lengths, sa_score_array, label='SA', marker='o')
    plt.plot(lengths, ga_score_array, label='GA', marker='o')
    plt.plot(lengths, mimic_score_array, label='MIMIC', marker='o')

    # Adding labels and title
    plt.xlabel('Problem size')
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend()

    # Display the plot
    plt.savefig(output_path)


def plot_size_variable_parallel(variable, length, rhc_score_dict, sa_score_dict, ga_score_dict, mimic_score_dict, title, y_lab, output_path):
    lengths = np.array(length)
    rhc_score_array = np.array([rhc_score_dict[key].get(variable) for key in range(len(rhc_score_dict))])
    sa_score_array = np.array([sa_score_dict[key].get(variable) for key in range(len(sa_score_dict))])
    ga_score_array = np.array([ga_score_dict[key].get(variable) for key in range(len(ga_score_dict))])
    mimic_score_array = np.array([mimic_score_dict[key].get(variable) for key in range(len(mimic_score_dict))])

    plt.figure(figsize=(10, 6))

    plt.plot(lengths, rhc_score_array, label='RHC ', marker='o')
    plt.plot(lengths, sa_score_array, label='SA', marker='o')
    plt.plot(lengths, ga_score_array, label='GA', marker='o')
    plt.plot(lengths, mimic_score_array, label='MIMIC', marker='o')

    # Adding labels and title
    plt.xlabel('Problem size')
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend()

    # Display the plot
    plt.savefig(output_path)

# helper functions
def get_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def put_pickle_file(filepath, file):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)


def encoder(df):
    labelencoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtypes=='object':
            df[col]=labelencoder.fit_transform(df[col])
    return df

def gen_save_data(final_samples=1500):
    df = pd.read_csv('src/section2/data/bank/train.csv')
    df = encoder(df)
    scaler = StandardScaler()

    X = df.drop(columns=['Exited', 'id','CustomerId', 'Surname'])
    X = scaler.fit_transform(X)
    y = df[['Exited']].astype('int')

    # define the undersampling method
    undersample = NearMiss(version=1, n_neighbors=3)
    X, y = undersample.fit_resample(X, y)

    # randomly sample rows to reduce data size
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    indices = indices[:final_samples]
    # Choose the same indices for both x and y
    X = X[indices]
    y = y.iloc[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = SEED_NUM) # (only used when there is only one dataset)
    put_pickle_file(X_TRAIN_PATH, X_train)
    put_pickle_file(X_TEST_PATH, X_test)
    put_pickle_file(Y_TRAIN_PATH, y_train)
    put_pickle_file(Y_TEST_PATH, y_test)
    return X_train, X_test, y_train, y_test

def load_dataset():
    X_train = get_hyperparameters(X_TRAIN_PATH)
    X_test = get_hyperparameters(X_TEST_PATH)
    y_train = get_hyperparameters(Y_TRAIN_PATH)
    y_test = get_hyperparameters(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test

def clean_array(array, max_len):
    if len(array) > max_len:
        return array[:max_len]
    elif len(array) < max_len:
        return np.pad(array, (0, max_len - len(array)), 'edge')
    else:
        return array
    
def trim_array(arr):
    max_index = np.argmax(arr)
    trimmed_arr = arr[:max_index + 1]
    return trimmed_arr