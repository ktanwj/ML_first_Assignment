from src.utils import *
from src.utils_ed import *

result = get_pickle_file('src/section1/hyperparameter tuning/rhc_results20_flipflop.pkl')
len(result)
# get_best_param(results, params)


rhc_grid = {
        "max_attempt": list(range(10, 300, 50)),
        "max_iter": [2000],
        "restart": list(range(0, 5, 1))
    }

generate_parameter_list(rhc_grid)