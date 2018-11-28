import inspect
from collections import OrderedDict

def get_names(sig):
    names = [(name, value) for name, value in sig.parameters.items()]
    return OrderedDict(names)


def filter_args(wanted_names, all_names, all_values):
    # pick out values corresponding to wanted_names in the correct order
    args = (all_values[all_names.index(name)] for name in wanted_names)
    return args


def skip_pars(names1, names2, num_skipped):
    skipped_pars1 = list(names1.keys())[:num_skipped]
    skipped_pars2 = list(names2.keys())[:num_skipped]
    if skipped_pars1 == skipped_pars2:
        pars1 = list(names1.values())[num_skipped:]
        pars2 = list(names2.values())[num_skipped:]
    else:
        raise Exception('First {} arguments '
                        'have to be the same'.format(num_skipped))
    return pars1, pars2


def combine(f, g, operator, num_skipped=0):
    if not callable(f) or not callable(g):
        raise Exception('One of the functions is not a function')

    sig1 = inspect.signature(f)
    sig2 = inspect.signature(g)

    names1 = get_names(sig1)
    names2 = get_names(sig2)

    pars1, pars2 = skip_pars(names1, names2, num_skipped)
    skipped_pars = list(names1.values())[:num_skipped]

    def wrapped(*args):
        # Only takes positional arguments, which is fine,
        # as kwant only ever calls value functions this way.
        args1 = filter_args(names1, all_names, args)
        args2 = filter_args(names2, all_names, args)

        fval = f(*args1)
        gval = g(*args2)
        return operator(fval, gval)

    pars1_names = [p.name for p in pars1]
    pars2 = [p for p in pars2 if p.name not in pars1_names]
    parameters = pars1 + pars2
    all_names = [p.name for p in skipped_pars + parameters]
    parameters = [p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in parameters]
    wrapped.__signature__ = inspect.Signature(parameters=skipped_pars + parameters)
    return wrapped
