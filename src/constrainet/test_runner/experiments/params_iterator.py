from typing import List
from benedict import benedict
from copy import deepcopy
from itertools import product
import re
from functools import lru_cache


def to_list_or_expanded_range(options) -> List:
    RANGE_SEP = ':'
    if isinstance(options, str):
        if RANGE_SEP in options:
            start, stop, *step = map(int, options.split(RANGE_SEP))
            return list(range(start, stop, step[0] if len(step) > 0 else 1))
        raise ValueError(f'Wrong options: {options!r}')
    return options


@lru_cache(2)
def chached_re_compile(pattern):
    return re.compile(pattern)


def string_interpolate(template: str, data: benedict, var_prefix: str):
    pattern = '\\' + var_prefix + '{([^}]*)}'
    regex = chached_re_compile(pattern)
    def _replace_action(match):
        key = match.group(1)
        if key not in data:
            raise ValueError(f'Invalid key for substitution: {key=!r} into template: {template!r}')
        return str(data[key])
    return regex.sub(_replace_action, template)


def expand_template(template, var_prefix: str = '$'):
    base_dict = deepcopy(template)
    q = []
    for k, v in template.items():
        q.append((v, k, base_dict[k]))

    variables = dict()
    strings_to_interpolate = dict()
    while len(q) > 0:
        node, path, base = q.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                new_path = path + '.' + k
                if isinstance(v, dict):
                    if k.startswith(var_prefix):
                        raise ValueError(f'Only leaf variables are allowed')
                    q.append((v, new_path, base[k]))
                else:
                    if k.startswith(var_prefix):
                        variables[path + '.' + k[len(var_prefix):]] = to_list_or_expanded_range(v)
                        del base[k]
                    elif isinstance(v, str) and var_prefix in v:
                        strings_to_interpolate[new_path] = v
            continue
        if isinstance(node, str) and var_prefix in node:
            strings_to_interpolate[path] = node

    for var_values in product(*variables.values()):
        cur_vars = dict(zip(variables.keys(), var_values))
        cur_dict = benedict(deepcopy(base_dict))
        for var_name, var_value in cur_vars.items():
            cur_dict[var_name] = var_value
        for s_name, s_template in strings_to_interpolate.items():
            cur_dict[s_name] = string_interpolate(s_template, cur_dict, var_prefix)
        yield cur_dict.dict()


def iterate_params(param_templates):
    for template in param_templates:
        yield from expand_template(template)


# if __name__ == '__main__':
#     templates = [
#         {
#             "name": "sgtc_parience=${solver.scheduler.patience}_factor=${solver.scheduler.factor}_cooldown=${solver.scheduler.cooldown}",
#             "some param": [123, 456],
#             "solver": {
#                 "n_solutions": 10,
#                 "optim": 'AdamW',
#                 "scheduler": {
#                     "$patience": "10:101:50",
#                     "$factor": [0.1, 0.2],
#                     "cooldown": 0,
#                     'test': '${solver.scheduler.patience},${solver.scheduler.factor}'
#                 }
#             }
#         },
#     ]
#     for params in iterate_params(templates):
#         print(benedict(params).dump())
