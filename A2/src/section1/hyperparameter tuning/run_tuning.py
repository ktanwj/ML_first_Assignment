from utils_ed import *
from utils import *

for length in list(range(10,100,10)):
    # flipflop
    problem = 'flipflop'
    p_flipflop = mlrose.DiscreteOpt(length = length, fitness_fn = problem_flipflop(), 
                    maximize = True, max_val = 2)
    p_knapsack = mlrose.DiscreteOpt(length = length, 
                                    fitness_fn = problem_knapsack(weights = list(range(2, length * 2 + 2, 2)),
                                                                values = list(range(1,length + 1,1)),
                                                                max_weight_pct = 0.6), 
                                    maximize = True, max_val = 2)
    rhc_grid = {
        "problem": [p_flipflop],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "restart": list(range(0, 5, 1)),
    }

    rhc_best_param = tune_params(rhc_grid, rhc_run, verbose = 10)
    rhc_results, rhc_wt = rhc_run(**rhc_best_param)

    with open(f'rhc_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(rhc_results, f)

    with open(f'rhc_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(rhc_wt, f)

    '''
    --------------------------------------------------------------------------------------------------------
    '''
    sa_grid = {
        "problem": [p_flipflop],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "decay": [mlrose.GeomDecay(init_temp=i, decay=j, min_temp=k) for i in list(range(5, 30, 5)) for j in [0.1, 0.3, 0.5, 0.7, 0.9] for k in [0.001, 0.01, 0.1, 0.5]]
    }

    sa_best_param = tune_params(sa_grid, sa_run, verbose = 10)
    sa_results, sa_wt = sa_run(**sa_best_param)

    with open(f'sa_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(sa_results, f)

    with open(f'sa_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(sa_wt, f)

    '''
    --------------------------------------------------------------------------------------------------------
    '''

    ga_grid = {
        "problem": [p_flipflop],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "pop_size": list(range(100, 400, 100)),
        "mutation_prob": [0.2, 0.5, 0.8]
    }

    ga_best_param = tune_params(ga_grid, ga_run, verbose = 10)
    ga_results, ga_wt = ga_run(**ga_best_param)

    with open(f'ga_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(ga_results, f)

    with open(f'ga_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(ga_wt, f)


    '''
    --------------------------------------------------------------------------------------------------------
    '''

    mimic_grid = {
        "problem": [p_flipflop],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "pop_size": list(range(100, 400, 100)),
        "keep_pct": [0.2, 0.5, 0.8]
    }

    mimic_best_param = tune_params(mimic_grid, mimic_run, verbose = 10)
    mimic_results, mimic_wt = mimic_run(**mimic_best_param)

    with open(f'mimic_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(mimic_results, f)

    with open(f'mimic_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(mimic_wt, f)

    '''
    KNAPSACK --------------------------------------------------------------------------------------------------------
    '''
    problem = 'knapsack'

    rhc_grid = {
        "problem": [p_knapsack],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "restart": list(range(0, 5, 1)),
    }

    rhc_best_param = tune_params(rhc_grid, rhc_run, verbose = 10)
    rhc_results, rhc_wt = rhc_run(**rhc_best_param)

    with open(f'rhc_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(rhc_results, f)

    with open(f'rhc_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(rhc_wt, f)

    '''
    --------------------------------------------------------------------------------------------------------
    '''
    sa_grid = {
        "problem": [p_knapsack],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "decay": [mlrose.GeomDecay(init_temp=i, decay=j, min_temp=k) for i in list(range(5, 30, 5)) for j in [0.1, 0.3, 0.5, 0.7, 0.9] for k in [0.001, 0.01, 0.1, 0.5]]
    }

    sa_best_param = tune_params(sa_grid, sa_run, verbose = 10)
    sa_results, sa_wt = sa_run(**sa_best_param)

    with open(f'sa_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(sa_results, f)

    with open(f'sa_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(sa_wt, f)

    '''
    --------------------------------------------------------------------------------------------------------
    '''

    ga_grid = {
        "problem": [p_knapsack],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "pop_size": list(range(100, 400, 100)),
        "mutation_prob": [0.2, 0.5, 0.8]
    }

    ga_best_param = tune_params(ga_grid, ga_run, verbose = 10)
    ga_results, ga_wt = ga_run(**ga_best_param)

    with open(f'ga_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(ga_results, f)

    with open(f'ga_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(ga_wt, f)


    '''
    --------------------------------------------------------------------------------------------------------
    '''

    mimic_grid = {
        "problem": [p_knapsack],
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "pop_size": list(range(100, 400, 100)),
        "keep_pct": [0.2, 0.5, 0.8]
    }

    mimic_best_param = tune_params(mimic_grid, mimic_run, verbose = 10)
    mimic_results, mimic_wt = mimic_run(**mimic_best_param)

    with open(f'mimic_results{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(mimic_results, f)

    with open(f'mimic_wt{length}_{problem}.pkl', 'wb') as f:
        pickle.dump(mimic_wt, f)