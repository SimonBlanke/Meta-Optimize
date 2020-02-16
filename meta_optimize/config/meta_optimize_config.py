# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .meta_opt_obj_func import hill_climbing
from .meta_opt_search_spaces import hill_climbing_search_space


hill_climbing_config = {hill_climbing: hill_climbing_search_space}


meta_opt_search_config = {"HillClimbing": hill_climbing_config}
