
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym

from frozenlake_reward import generate_random_map
from test_env_custom import TestEnv
from utils import v_iters_plot, plot_v_iters_per_state, values_heat_map_with_env, values_heat_map

# from gym.envs.toy_text.frozen_lake import generate_random_map
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL

FOLDER_PATH = 'src/bj_output/PI/'

from bettermdptools.utils.decorators import add_to

@add_to(Plots)
@staticmethod
def modified_plot_policy(val_max, directions, map_size, title):
    """Plot the policy learned."""
    sns.heatmap(
        val_max,
        annot=directions,
        fmt="",
        cmap=sns.color_palette("magma_r", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
    ).set(title=title)
    img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
    plt.savefig(FOLDER_PATH + f'policy_map_g{gamma}.png')
    plt.close()


"""
For every gamma, plot
    1. avg steps
    2. success rate (number of goal reached)
    3. average reward
    4. time to converge
"""
# initialisation
N_ITERS = 1000000
TEST_ITERS = 1000

GAMMA_LIST = np.arange(0.1, 1.1, 0.1)

iter_results_all = {}
success_results_all = {}
reward_results_all = {}
runtime_results_all = {}

iter_results = []
success_results = []
reward_results = []
runtime_results = []


base_env = gym.make('Blackjack-v1',
                    natural=False,
                    sab=False,
                    render_mode = None)
blackjack = BlackjackWrapper(base_env)

for gamma in GAMMA_LIST:
    start = time.time()
    V, V_track, pi = Planner(blackjack.P).policy_iteration(gamma = gamma, n_iters=N_ITERS)
    end = time.time()
    time_to_converge = end - start
    testscores, success_steps, successes = TestEnv.test_env(blackjack, pi = pi, n_iters = TEST_ITERS)
    
    title = f"Blackjack\nPolicy Map for Gamma:{gamma}"
    #create actions dictionary and set map size
    blackjack_actions = {0: "S", 1: "H"}
    blackjack_map_size=(29, 10)
    #get formatted state values and policy map
    val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)
    Plots.modified_plot_policy(val_max, policy_map, blackjack_map_size, title)

    # save results
    iter_results.append(round(np.mean(success_steps), 2))
    success_results.append(round(np.mean(successes), 2)*100)
    reward_results.append(round(np.mean(testscores), 2))
    runtime_results.append(round(time_to_converge, 2))

    values_heat_map(V, "Blackjack\nValue Iteration State Values", blackjack_map_size, FOLDER_PATH + f'value_map_g{gamma}.png')


plt.figure(figsize=(16, 16))
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot data on each subplot
axs[0, 0].plot(GAMMA_LIST, reward_results)

axs[0, 0].set_xlabel('Gamma value')
axs[0, 0].set_ylabel('Average Reward')
axs[0, 0].set_title(f'Average Reward vs. Gamma (VI)')

axs[0, 1].plot(GAMMA_LIST, success_results)

axs[0, 1].set_xlabel('Gamma value')
axs[0, 1].set_ylabel('Average Success Rate (%)')
axs[0, 1].set_title(f'Average Success Rate (%) vs. Gamma (VI)')

axs[1, 0].plot(GAMMA_LIST, iter_results)

axs[1, 0].set_xlabel('Gamma value')
axs[1, 0].set_ylabel('Average Steps')
axs[1, 0].set_title(f'Average Steps vs. Gamma (VI)')

axs[1, 1].plot(GAMMA_LIST, runtime_results)

axs[1, 1].set_xlabel('Gamma value')
axs[1, 1].set_ylabel('Average Runtime')
axs[1, 1].set_title(f'Average Runtime vs. Gamma (VI)')

plt.tight_layout()
plt.savefig(FOLDER_PATH + f'v_iters_raw_plot.png')