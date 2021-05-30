from xsimlab.model import _ModelBuilder, filter_variables
from xsimlab.variable import VarType
import numpy as np

from .vmlab import (
    create_setup,
    run,
    get_vars_from_model,
    load_graph,
    check_graph
)
from . import constants, enums
from .vmlab import DotDict


def fill_value_from_dtype(dtype=None):
    """ Try to keep NaN for not yet appeard or pruned GUs """
    if dtype is None:
        return 0.
    if dtype.kind in 'f':
        return 0.
    elif dtype.kind == 'M':
        return np.datetime64('NaT')
    elif dtype.kind == 'O':
        return None
    else:
        return 0.


class State(dict):

    indices = {}
    variables = {}

    def __init__(self, variables, indices):
        self.variables = variables
        self.indices = indices
        dict.__init__(self)

    def resize(self, index_name, new_shape):
        '''Resize a variable value if a 1D index  in the variables dimensions increased in length.
        '''
        for var_name, var in self.variables.items():
            if var_name in self and type(self[var_name]) is np.ndarray:
                var_value = self[var_name]
                var_shape = var_value.shape
                var_dims = np.array(var.metadata.get('dims')).flatten()
                if len(var_dims) and index_name in var_dims:
                    fill_value = None
                    var_enc = var.metadata.get('encoding')
                    if 'fill_value' in var_enc:
                        fill_value = var_enc['fill_value']
                    else:
                        fill_value = fill_value_from_dtype(var_value.dtype)
                    var_shape = (var_dims == index_name) * new_shape + (var_dims != index_name) * var_shape
                    if (tuple(var_shape) != var_value.shape):
                        if len(var_shape) == 1:
                            data = np.empty(var_shape, dtype=var_value.dtype)
                            data[0:var_value.shape[0]] = var_value
                            data[var_value.shape[0]:] = fill_value
                        else:
                            data = np.full(var_shape, fill_value, dtype=var_value.dtype)
                            data[tuple([slice(0, i) for i in var_value.shape])] = var_value
                        super(State, self).__setitem__(var_name, data)

    def __setitem__(self, item, new):
        if item in self and item in self.indices and self[item].shape != new.shape:
            self.resize(item[1], new.shape)
        super(State, self).__setitem__(item, new)


def set_state(self):
    variables = {}
    indices = {}
    non_indices = {}
    for p_name, p_obj in self._processes_obj.items():
        for v_name, variable in filter_variables(p_obj).items():
            if variable.metadata['var_type'] == VarType.INDEX:
                indices[(p_name, v_name)] = variable
            else:
                non_indices[(p_name, v_name)] = variable
            variables[(p_name, v_name)] = variable

    state = State(non_indices, indices)

    # bind state to each process in the model
    for p_obj in self._processes_obj.values():
        p_obj.__xsimlab_state__ = state

    return state


# 'patch' xsimlab to use a custom state class instead of a dict
_ModelBuilder.set_state = set_state


__all__ = [
    'create_setup',
    'run',
    'constants',
    'enums',
    'DotDict',
    'get_vars_from_model',
    'load_graph',
    'check_graph'
]
