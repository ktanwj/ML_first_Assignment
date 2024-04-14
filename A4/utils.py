import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle

# parallel computing packages
from itertools import product
from joblib import Parallel, delayed

# from gym.envs.toy_text.frozen_lake import generate_random_map
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL

from tqdm import tqdm

def v_iters_plot(data, line_val, title, save_path):
    df = pd.DataFrame(data=data)
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, legend=None).set_title(title)
    plt.axvline(x=line_val, color='r', linestyle='--')
    plt.savefig(save_path)
    plt.close()


def plot_v_iters_per_state(V_track, n_state):
    max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')

    # find convergence iter
    convergence_iter = V_track.shape[1]
    max_avg_v = np.max(np.trim_zeros(np.mean(V_track, axis=1)))
    theta = 0.5/100*max_avg_v
    for i in range(V_track.shape[0]-1):
        if np.abs(np.mean(V_track[i]) - max_avg_v) < theta:
            convergence_iter = i
            break
    # plot v_iters
    v_iters_plot(max_value_per_iter,
                       convergence_iter,
                       f"Frozen Lake\nAverage Value vs. Iterations for state {n_state}",
                       save_path = f'./src/fl_output/v_iters_{n_state}.png')
    


def values_heat_map_with_env(env, data, title, size, FILE_PATH):
    grid = env.desc.astype(str)
    grid[grid == 'S'] = '0'  # Start state
    grid[grid == 'G'] = '1'  # Goal state
    grid[grid == 'F'] = '2'  # Frozen state (safe)
    grid[grid == 'H'] = '3'  # Hole state (unsafe)
    
    data = np.around(np.array(data).reshape(size), 2)
    df = pd.DataFrame(data=data)
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=False, ax=ax, cbar=True)
    ax.imshow(np.ones_like(data), alpha=0, extent=ax.get_xlim() + ax.get_ylim())

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i, j] == '3':  # Hole state
                ax.text(j + 0.5, i + 0.5, 'H', color='white', ha='center', va='center')
            elif grid[i, j] == '1':
                ax.text(j + 0.5, i + 0.5, 'G', color='white', ha='center', va='center')
            else:
                ax.text(j + 0.5, i + 0.5, data[i, j], color='black', ha='center', va='center')
    
    ax.set_title(title)
    plt.savefig(FILE_PATH)
    plt.close()


# helper functions
def get_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def put_pickle_file(filepath, file):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)

def generate_parameter_list(parameter_dict):
    combinations = product(*parameter_dict.values())
    parameters_list = [
        dict(zip(parameter_dict.keys(), combination)) for combination in combinations
    ]

    return parameters_list

def values_heat_map(data, title, size, FILE_PATH):
        data = np.around(np.array(data).reshape(size), 2)
        df = pd.DataFrame(data=data)
        sns.heatmap(df, annot=True).set_title(title)
        plt.savefig(FILE_PATH)
        plt.close()

@staticmethod
def get_policy_map(pi, val_max, actions, map_size):
    """Map the best learned action to arrows."""
    #convert pi to numpy array
    best_action = np.zeros(val_max.shape[0], dtype=np.int32)
    for idx, val in enumerate(val_max):
        best_action[idx] = pi[idx]
    policy_map = np.empty(best_action.flatten().shape, dtype=str)
    for idx, val in enumerate(best_action.flatten()):
        policy_map[idx] = actions[val]
    policy_map = policy_map.reshape(map_size[0], map_size[1])
    val_max = val_max.reshape(map_size[0], map_size[1])
    return val_max, policy_map

#modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
@staticmethod
def plot_policy(val_max, directions, map_size, title, FILE_PATH):
    """Plot the policy learned."""
    sns.heatmap(
        val_max,
        annot=directions,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title=title)
    img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
    plt.savefig(FILE_PATH)
    plt.close()