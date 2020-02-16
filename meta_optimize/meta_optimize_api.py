# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import numpy as np
from hyperactive import Hyperactive


from .config.meta_optimize_config import meta_opt_search_config


X, y = np.array([0]), np.array([0])


class MetaOptimize:
    def __init__(self,):
        self.opt = Hyperactive(X, y, memory="short", verbosity=3)

    def optimize(
        self,
        optimizer,
        hyperactive_config={
            "n_iter": 100,
            "max_time": None,
            "optimizer": "RandomSearch",
            "n_jobs": 1,
            "init_config": None,
        },
    ):

        search_config = meta_opt_search_config[optimizer]
        self.opt.search(search_config, **hyperactive_config)

        model = list(search_config.keys())[0]
        best_para = self.opt.results[model]
        score = self.opt.best_scores[model]

        best_opt_para = {optimizer: best_para}
        # print("\nbest_opt_para =", best_opt_para)
        # print("score         =", score)
        return best_opt_para
