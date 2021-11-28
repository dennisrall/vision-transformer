import yaml

from config import params_file


def get_params(*keys):
    with open(params_file) as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
    for key in keys:
        params = params[key]
    return params


def prod(iterable):
    res = 1
    for item in iterable:
        res *= item
    return res


def reverse_dict_order(d: dict) -> dict:
    tuples = [(key, value) for key, value in d.items()]
    return dict(reversed(tuples))
