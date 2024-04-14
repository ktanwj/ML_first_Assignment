import numpy as np
import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.rl import RL

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stepped = []
        self.max_episode_steps = 10000

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # Modify reward here as you see fit
        if terminated and reward == 1:
            reward =  1
        # If the agent falls into a hole, give a high negative reward
        elif terminated and reward == 0:
            reward = -1
        # Otherwise, give a reward proportional to the distance to the goal
        else:
            if [state // self.env.ncol, state % self.env.ncol] in self.stepped:
                reward = 0
            else:
                # Get the row and column indices of the current state
                row, col = state // self.env.ncol, state % self.env.ncol
                # Get the row and column indices of the goal state
                goal_row, goal_col = self.env.nrow - 1, self.env.ncol - 1
                # Calculate the Manhattan distance from the current state to the goal state
                distance_to_goal = abs(goal_row - row) + abs(goal_col - col)
                # Scale the reward based on the distance to the goal
                reward = 1 / (distance_to_goal + 1) 

                self.stepped.append([row, col])
        mod_reward = reward

        # Return modified reward with the rest of information
        return state, mod_reward, terminated, truncated, info


# with a wrapper
np.random.seed(0)
frozen_lake = gym.make('FrozenLake-v1', desc=generate_random_map(size=16), render_mode='ansi')
new_frozen_lake = RewardShapingWrapper(frozen_lake)

Q, V, pi, Q_track, pi_track = RL(new_frozen_lake).q_learning(init_alpha = 0.1, min_alpha = 0.05, n_episodes = 100000, epsilon_decay_ratio = 0.9)
frozen_lake = gym.make('FrozenLake-v1', desc=generate_random_map(size=16), render_mode='ansi')
test_scores = TestEnv.test_env(env=frozen_lake, n_iters=1000, render=False, pi=pi, user_input=False)
print(np.mean(test_scores))