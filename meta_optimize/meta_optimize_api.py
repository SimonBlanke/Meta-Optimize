# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import numpy as np
from hyperactive import Hyperactive


from .config.meta_optimize_config import meta_opt_search_config


X, y = np.array([0]), np.array([0])


class MetaOptimize:
    def __init__(
        self,
        n_iter=10,
        max_time=None,
        optimizer="RandomSearch",
        n_jobs=1,
        init_config=None,
    ):
        self.opt = Hyperactive(X, y, memory="short", verbosity=3)

        self.n_iter = n_iter
        self.max_time = max_time
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.init_config = init_config

    def optimize(self, to_optimize):

        search_config = meta_opt_search_config[to_optimize]
        self.opt.search(
            search_config,
            n_iter=self.n_iter,
            max_time=self.max_time,
            optimizer=self.optimizer,
            n_jobs=self.n_jobs,
            init_config=self.init_config,
        )

        model = list(search_config.keys())[0]
        best_para = self.opt.results[model]
        score = self.opt.best_scores[model]

        best_opt_para = {to_optimize: best_para}
        # print("\nbest_opt_para =", best_opt_para)
        # print("score         =", score)
        return best_opt_para
