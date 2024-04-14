import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym


from frozenlake_reward import generate_random_map, FrozenLakeEnv
from test_env_custom import TestEnv
from utils import v_iters_plot, plot_v_iters_per_state, values_heat_map_with_env, plot_policy
from tqdm import tqdm 

from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL

FOLDER_PATH = 'src/fl_output/PI_SHAPE/'

"""
Same plot but using reward shaping
"""

# initialisation
N_ITERS = 10000
TEST_ITERS = 100
G_REWARD = 100
N_STATE_LIST = [4, 6, 8, 10, 12, 16, 20]
GAMMA_LIST = np.arange(0.1, 1.1, 0.1)
SEED = 40

iter_results_all = {}
success_results_all = {}
reward_results_all = {}
runtime_results_all = {}

for n_state in tqdm(N_STATE_LIST):
    print(f'current state: {n_state}')
    iter_results = []
    success_results = []
    reward_results = []
    runtime_results = []
    frozen_lake = FrozenLakeEnv(desc=generate_random_map(size=n_state, p=0.9, seed=SEED), g_reward = G_REWARD, render_mode=None)

    for gamma in GAMMA_LIST:
        start = time.time()
        V, V_track, pi = Planner(frozen_lake.P).policy_iteration(gamma = gamma, n_iters=N_ITERS)
        end = time.time()
        time_to_converge = end - start
        testscores, success_steps, successes = TestEnv.test_env(frozen_lake, goal_reward=G_REWARD, pi = pi, n_iters = TEST_ITERS)

        # plot heat map
        if n_state < 10:
            values_heat_map_with_env(frozen_lake, V, f"FL\nValue Iteration State Value for Gamma:{gamma}, State:{n_state}", size=(n_state,n_state), FILE_PATH=FOLDER_PATH + f'heatmap/v_heatmap_g{gamma}_s{n_state}.png')

            fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
            fl_map_size=(n_state,n_state)
            title=f"FL Mapped Policy for Gamma:{gamma}, State: {n_state}\nArrows represent best action"
            val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)
            plot_policy(val_max, policy_map, fl_map_size, title, FILE_PATH=FOLDER_PATH + f'policymap/policy_map_g{gamma}_s{n_state}.png')

        iter_results.append(round(np.mean(success_steps), 2))
        success_results.append(round(np.mean(successes), 2)*100)
        reward_results.append(round(np.mean(testscores), 2))
        runtime_results.append(round(time_to_converge, 2))

    iter_results_all[n_state] = iter_results
    success_results_all[n_state] = success_results
    reward_results_all[n_state] = reward_results
    runtime_results_all[n_state] = runtime_results
    
    plot_v_iters_per_state(V_track, n_state)

plt.figure(figsize=(16, 16))
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot data on each subplot
for state, values in reward_results_all.items():
    axs[0, 0].plot(GAMMA_LIST, values, label=f'State {state}')

axs[0, 0].set_xlabel('Gamma value')
axs[0, 0].set_ylabel('Average Reward')
axs[0, 0].set_title(f'Average Reward vs. Gamma (VI)')
axs[0, 0].legend(loc="upper left", fontsize='small')

for state, values in success_results_all.items():
    axs[0, 1].plot(GAMMA_LIST, values, label=f'State {state}')

axs[0, 1].set_xlabel('Gamma value')
axs[0, 1].set_ylabel('Average Success Rate (%)')
axs[0, 1].set_title(f'Average Success Rate (%) vs. Gamma (VI)')
axs[0, 1].legend(loc="upper left", fontsize='small')

for state, values in iter_results_all.items():
    axs[1, 0].plot(GAMMA_LIST, values, label=f'State {state}')

axs[1, 0].set_xlabel('Gamma value')
axs[1, 0].set_ylabel('Average Steps')
axs[1, 0].set_title(f'Average Steps vs. Gamma (VI)')
axs[1, 0].legend(loc="upper left", fontsize='small')

for state, values in runtime_results_all.items():
    axs[1, 1].plot(GAMMA_LIST, values, label=f'State {state}')

axs[1, 1].set_xlabel('Gamma value')
axs[1, 1].set_ylabel('Average Runtime')
axs[1, 1].set_title(f'Average Runtime vs. Gamma (VI)')
axs[1, 1].legend(loc="upper left", fontsize='small')

plt.tight_layout()
plt.savefig(FOLDER_PATH + f'v_iters_reward_plot.png')