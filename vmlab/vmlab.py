import xsimlab as xs
from xsimlab.variable import VarIntent
import pandas as pd
import numpy as np
import igraph as ig
import io
import toml
import warnings
import pathlib


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_topology_inputs_from_df(topology_prc, df, cycle):
    input_topo = {}
    required_attrs = ['id', 'parent_id', 'cycle', 'is_apical', 'appearance_month', 'ancestor_nature', 'ancestor_is_apical']
    assert len(set(df.columns) & set(required_attrs)) == len(required_attrs)

    df = df[df['cycle'] <= cycle]
    edges = df[['id', 'parent_id']][pd.notna(df['parent_id'])]
    vertices = df[['id', 'cycle', 'is_apical', 'appearance_month', 'ancestor_nature', 'ancestor_is_apical']]

    graph = ig.Graph.DictList(
        vertices=vertices.to_dict('records'),
        edges=edges.to_dict('records'),
        vertex_name_attr='id',
        edge_foreign_keys=('parent_id', 'id'),
        directed=True
    )
    assert graph.is_dag()

    for var_name in xs.filter_variables(topology_prc, var_type='variable', func=lambda var: var.metadata['intent'] == VarIntent.INOUT):
        if var_name in graph.vs.attribute_names():
            input_topo[f'topology__{var_name}'] = np.array(graph.vs.get_attribute_values(var_name), dtype=np.float32)
    input_topo['topology__adjacency'] = np.array(graph.get_adjacency().data, dtype=np.float32)

    return input_topo, graph


def create_setup(
    model,
    start_date,
    end_date,
    current_cycle,
    clocks={},
    initial_tree_df=None,
    input_vars=None,
    output_vars=None,
    fill_default=True,
    setup_toml=None
):

    input_vars = {
        **({} if input_vars is None else input_vars)
    }

    main_clock = 'day'
    clocks = {} if clocks is None else clocks
    clocks[main_clock] = pd.date_range(start=start_date, end=end_date, freq='1d')

    # set toml file path as process input from 'parameters' section in setup_toml
    if setup_toml is not None:
        with io.open(setup_toml) as setup_file:
            setup = toml.loads(setup_file.read())
            dir_path = pathlib.Path(setup_toml).parent
            if 'parameters' in setup:
                for prc_name, rel_file_path in setup['parameters'].items():
                    path = dir_path.joinpath(rel_file_path)
                    if prc_name in model:
                        if path.exists():
                            # process 'prc_name' must inherit from ParameterizedProcess or
                            # declare a parameter_file_path 'in' variable and handle it
                            input_vars[f'{prc_name}__parameter_file_path'] = str(path)
                        else:
                            warnings.warn(f'Input file "{path}" does not exist')
            if 'initial_tree' in setup and initial_tree_df is None:
                if not setup['initial_tree']:
                    raise ValueError('No iniyial tree provided')
                else:
                    path = dir_path.joinpath(setup['initial_tree'])
                    initial_tree_df = pd.read_csv(path).astype(np.float32)

    graph = None
    if 'topology' in model:
        input_topo, graph = get_topology_inputs_from_df(model['topology'], initial_tree_df, current_cycle)
        input_vars.update(input_topo)
        input_vars['topology__current_cycle'] = current_cycle
        # work-around for main_clock not available at initialization.
        # set the start date variable
        input_vars['topology__sim_start_date'] = start_date

    output_vars_ = {}
    for prc_name in model:
        output_vars_[prc_name] = {}
        prc = model[prc_name]
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: var.metadata['static']):
            output_vars_[prc_name][var_name] = None
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: not var.metadata['static']):
            output_vars_[prc_name][var_name] = output_vars if type(output_vars) is str else None  # str must be clock name
        if graph is not None:
            # make simlab happy by passing initial 'inout' values (needlessly)
            shape = (len(graph.vs.indices),)
            for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: var.metadata['intent'] == VarIntent.INOUT and var.metadata['dims'] == (('GU',),)):
                if f'{prc_name}__{var_name}' not in input_vars:
                    input_vars[f'{prc_name}__{var_name}'] = np.empty(shape, dtype=np.float32)

    return xs.create_setup(
        model,
        clocks,
        main_clock,
        input_vars,
        output_vars_,
        fill_default
    )
