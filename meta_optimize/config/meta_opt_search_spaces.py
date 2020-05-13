# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from numpy.random import normal, laplace, logistic, gumbel


hill_climbing_search_space = {
    "epsilon": np.arange(0.01, 3, 0.01),
    "distribution": [normal, laplace, logistic, gumbel],
    "n_neighbours": range(1, 11),
}
