import numpy as np


def create_search_space(size, dim):
    _dict_ = {}
    for i in range(dim):
        _dict_[str(i)] = size

    return _dict_


def sphere_function(para, X, y):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue
        loss.append(para[key] * para[key])

    return -np.array(loss).sum()


def rastrigin_function(para, X, y):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue

        loss_1d = 1 + para[key] * para[key] - np.cos(2 * np.pi * para[key])
        loss.append(loss_1d)

    return -(np.array(loss).sum())


def ackley_function(para, X, y):
    x, y = para["x"], para["y"]

    loss = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) + np.exp(1) + 20
    )

    return -loss


def rosenbrock_function(para, X, y):
    x = np.array([para["x1"], para["x2"], para["x3"], para["x4"]])
    y = np.array([para["x0"], para["x1"], para["x2"], para["x3"]])

    loss = sum(100.0 * (x - y ** 2.0) ** 2.0 + (1 - y) ** 2.0)

    return -loss


def himmelblau_function(para, X, y):
    loss = (para["x"] ** 2 + para["y"] - 11) ** 2 + (
        para["x"] + para["y"] ** 2 - 7
    ) ** 2

    return -loss


dim_size = np.arange(-10, 10, 0.03)


sphere_function_search_config_3 = {sphere_function: create_search_space(dim_size, 3)}


rastrigin_function_search_config_3 = {
    rastrigin_function: create_search_space(dim_size, 3)
}

sphere_function_search_config_5 = {sphere_function: create_search_space(dim_size, 5)}


rastrigin_function_search_config_5 = {
    rastrigin_function: create_search_space(dim_size, 5)
}


sphere_function_search_config_7 = {sphere_function: create_search_space(dim_size, 7)}


rastrigin_function_search_config_7 = {
    rastrigin_function: create_search_space(dim_size, 7)
}

sphere_function_search_config_9 = {sphere_function: create_search_space(dim_size, 9)}


rastrigin_function_search_config_9 = {
    rastrigin_function: create_search_space(dim_size, 9)
}

ackley_function_search_config = {ackley_function: {"x": dim_size, "y": dim_size}}

rosenbrock_function_search_config = {
    rosenbrock_function: {
        "x0": dim_size,
        "x1": dim_size,
        "x2": dim_size,
        "x3": dim_size,
        "x4": dim_size,
    }
}

himmelblau_function_search_config = {
    himmelblau_function: {"x": dim_size, "y": dim_size}
}


test_func_search_configs = [
    sphere_function_search_config_3,
    rastrigin_function_search_config_3,
    sphere_function_search_config_5,
    rastrigin_function_search_config_5,
    sphere_function_search_config_7,
    rastrigin_function_search_config_7,
    sphere_function_search_config_9,
    rastrigin_function_search_config_9,
    ackley_function_search_config,
    rosenbrock_function_search_config,
    himmelblau_function_search_config,
]
