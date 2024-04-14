# parallel computing packages
from itertools import product
from joblib import Parallel, delayed

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym

from frozenlake_reward import generate_random_map, FrozenLakeEnv
from test_env_custom import TestEnv
from utils import plot_v_iters_per_state, values_heat_map_with_env, generate_parameter_list, put_pickle_file

from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL

FOLDER_PATH = 'src/fl_output/QL_SHAPE/'


rhc_grid = {
        "gamma": list(np.arange(0.8, 1.1, 0.1)),
        "epsilon": list(np.arange(0.8, 1.1, 0.1)),
        "alpha": [0.4, 0.6, 0.8, 1.0],
        "epsilon_decay_ratio": [0.8, 0.9, 1.0],
        "alpha_decay_ratio": [0.8, 0.9, 1.0],
        "frozen_lake": [FrozenLakeEnv(desc=generate_random_map(size=8, p=0.9, seed=40), g_reward = 100, render_mode=None)]
    }

def run_ql(gamma, epsilon, alpha, epsilon_decay_ratio, alpha_decay_ratio, frozen_lake, G_REWARD = 100, TEST_ITERS = 50, N_ITERS=100000):
    start = time.time()
    Q, V, pi, Q_track, pi_track = RL(frozen_lake).q_learning(gamma = gamma,
                                                             init_epsilon = epsilon,
                                                             init_alpha = alpha,
                                                             alpha_decay_ratio = alpha_decay_ratio,
                                                             epsilon_decay_ratio = epsilon_decay_ratio,
                                                             n_episodes=N_ITERS)
    end = time.time()
    time_to_converge = end - start
    output = [Q, V, pi, Q_track, pi_track, time_to_converge]
    put_pickle_file(f"fl_ql_{gamma}_{epsilon}_{alpha}_{epsilon_decay_ratio}_{alpha_decay_ratio}.pkl", output)

parameters_list = generate_parameter_list(rhc_grid)
Parallel(n_jobs=-1, verbose=0)(delayed(run_ql)(**params) for params in parameters_list)