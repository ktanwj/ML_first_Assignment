from src.utils import *
from src.utils_ed import *

def plot_feval_iteration(max_iteration, problem_size):
    problem_list = ['flipflop', 'fourpeak', 'knapsack']

    for problem in problem_list:
        rhc_score_dict = get_pickle_file(f'src/section1/{problem}/rhc_experiment_dict.pkl')
        sa_score_dict = get_pickle_file(f'src/section1/{problem}/sa_experiment_dict.pkl')
        ga_score_dict = get_pickle_file(f'src/section1/{problem}/ga_experiment_dict.pkl')
        mimic_score_dict = get_pickle_file(f'src/section1/{problem}/mimic_experiment_dict.pkl')
        problem_dict = {10:0,
                        20:1,
                        30:2,
                        40:3,
                        50:4,
                        60:5,
                        70:6,
                        80:7,
                        90:8,
                        100:9}
        problem_index = problem_dict[problem_size]

        iteration = [i for i in range(max_iteration)]
 
        rhc_feval_50 = rhc_score_dict[problem_index]['best_fitness_curve'][:, 1]
        sa_feval_50 = sa_score_dict[problem_index]['best_fitness_curve'][:, 1]
        ga_feval_50 = ga_score_dict[problem_index]['best_fitness_curve'][:, 1]
        mimic_feval_50 = mimic_score_dict[problem_index]['best_fitness_curve'][:, 1]

        rhc_feval_50 = trim_array(rhc_feval_50)
        sa_feval_50 = trim_array(sa_feval_50)
        ga_feval_50 = trim_array(ga_feval_50)
        mimic_feval_50 = trim_array(mimic_feval_50)

        # Plotting the data
        plt.clf()
        plt.plot([i for i in range(len(rhc_feval_50))], rhc_feval_50, label = 'RHC', linewidth=2)
        plt.plot([i for i in range(len(sa_feval_50))], sa_feval_50, label = 'SA', linewidth=2)
        plt.plot([i for i in range(len(ga_feval_50))], ga_feval_50, label = 'GA', linewidth=2)
        plt.plot([i for i in range(len(mimic_feval_50))], mimic_feval_50, label = 'MIMIC', linewidth=2)
        
        # Adding labels and title
        plt.xlabel('Number of Iterations')
        plt.ylabel('Fitness Evaluations')
        plt.title('Fitness Evaluations vs. Number of Iterations')
        plt.legend()
        
        # Display the plot
        plt.savefig(f'src/section1/{problem}/feval_iteration_{problem_size}.png')


def plot_fitness_iteration(max_iteration, problem_size):
    problem_list = ['flipflop', 'fourpeak', 'knapsack']

    for problem in problem_list:
        rhc_score_dict = get_pickle_file(f'src/section1/{problem}/rhc_experiment_dict.pkl')
        sa_score_dict = get_pickle_file(f'src/section1/{problem}/sa_experiment_dict.pkl')
        ga_score_dict = get_pickle_file(f'src/section1/{problem}/ga_experiment_dict.pkl')
        mimic_score_dict = get_pickle_file(f'src/section1/{problem}/mimic_experiment_dict.pkl')
        problem_dict = {10:0, 20:1, 30:2, 40:3,
                        50:4, 60:5, 70:6, 80:7,
                        90:8, 100:9}
        problem_index = problem_dict[problem_size]

        

        rhc_fitness_curve_50 = rhc_score_dict[problem_index]['best_fitness_curve'][:, 0]
        sa_fitness_curve_50 = sa_score_dict[problem_index]['best_fitness_curve'][:, 0]
        ga_fitness_curve_50 = ga_score_dict[problem_index]['best_fitness_curve'][:, 0]
        mimic_fitness_curve_50 = mimic_score_dict[problem_index]['best_fitness_curve'][:, 0]
        

        rhc_fitness_curve_50 = trim_array(rhc_fitness_curve_50)
        sa_fitness_curve_50 = trim_array(sa_fitness_curve_50)
        ga_fitness_curve_50 = trim_array(ga_fitness_curve_50)
        mimic_fitness_curve_50 = trim_array(mimic_fitness_curve_50)

        # Plotting the data
        plt.clf()
        plt.plot([i for i in range(len(rhc_fitness_curve_50))], rhc_fitness_curve_50, label = 'RHC', linewidth=2)
        plt.plot([i for i in range(len(sa_fitness_curve_50))], sa_fitness_curve_50, label = 'SA', linewidth=2)
        plt.plot([i for i in range(len(ga_fitness_curve_50))], ga_fitness_curve_50, label = 'GA', linewidth=2)
        plt.plot([i for i in range(len(mimic_fitness_curve_50))], mimic_fitness_curve_50, label = 'MIMIC', linewidth=2)
        
        # Adding labels and title
        plt.xlabel('Number of Iterations')
        plt.ylabel('Fitness Score')
        plt.title('Fitness Scores vs. Number of Iterations')
        plt.legend()

        # Display the plot
        plt.savefig(f'src/section1/{problem}/fitness_iteration{problem_size}.png') 


plot_feval_iteration(max_iteration=1000, problem_size=50)
plot_feval_iteration(max_iteration=1000, problem_size=90)
plot_fitness_iteration(max_iteration=1000, problem_size=50)
plot_fitness_iteration(max_iteration=1000, problem_size=90)