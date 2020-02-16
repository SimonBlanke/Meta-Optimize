# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from hyperactive import Hyperactive
from .test_functions import test_func_search_configs


n_iters = [10, 25, 33, 50, 75, 100]
runs = range(10)


def hill_climbing(para, X, y):
    optimizer_config = {
        "HillClimbing": {
            "epsilon": para["epsilon"],
            "climb_dist": para["climb_dist"],
            "n_neighbours": para["n_neighbours"],
        }
    }

    loss_opt = []
    for i in runs:
        for n_iter in n_iters:
            for search_config in test_func_search_configs:
                opt = Hyperactive(X, y, memory="short", random_state=i, verbosity=0)
                opt.search(search_config, n_iter=n_iter, optimizer=optimizer_config)

                loss_opt.append(opt.best_scores[list(search_config.keys())[0]])

    score = np.array(loss_opt).mean()

    return score
