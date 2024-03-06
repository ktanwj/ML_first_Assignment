from src.utils import *
from src.utils_ed import *
import six
import sys
sys.modules['sklearn.externals.six'] = six

import mlrose_hiive as mlrose
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import NearMiss
import numpy as np
import pandas as pd


# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# sklearn models
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from __future__ import print_function

import warnings
import time
warnings.filterwarnings("ignore")
%matplotlib inline
SEED_NUM = 42

# CONFIG
X_TRAIN_PATH = 'src/section2/data/bank/x_train.pkl'
X_TEST_PATH = 'src/section2/data/bank/x_test.pkl'
Y_TRAIN_PATH = 'src/section2/data/bank/y_train.pkl'
Y_TEST_PATH = 'src/section2/data/bank/y_test.pkl'
    
def tuning(algo):
    X_train, X_test, y_train, y_test = load_dataset()
    learning_rate_param = [0.01, 0.05, 0.1]
    attempt_param = [50, 100]
    restart_param = [0, 3]

    param_dict = {"learning_rate": learning_rate_param,
                  "attempt": attempt_param,
                  "restart": restart_param}
    all_results = []
    param_list = generate_parameter_list(param_dict)

    for param in param_list:
        result = {}
        learning_rate = param["learning_rate"]
        attempt = param["attempt"]
        restart = param["restart"]

        nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu',
                                                        algorithm=algo, max_iters=1000,
                                                        bias=True, is_classifier=True, learning_rate=learning_rate,
                                                        early_stopping=False, clip_max=5, max_attempts=attempt,
                                                        restarts=restart, curve=True)
        nn_model_rhc.fit(X_train,y_train)
        y_pred = nn_model_rhc.predict(X_test).flatten()
        result['algo'] = algo
        result['learning_rate'] = learning_rate
        result['attempt'] = attempt
        result['restart'] = restart
        result['fitness_curve'] = nn_model_rhc.fitness_curve
        result['f1_score'] = f1_score(y_test, y_pred)

        all_results.append(result)

hyperparam_results = []

hyperparam_results = Parallel(n_jobs=-1)(delayed(tuning)(algo) for algo in ['random_hill_climb', 'simulated_annealing', 'genetic_alg'])
put_pickle_file('src/section2/hyperparam_results.pkl', hyperparam_results)