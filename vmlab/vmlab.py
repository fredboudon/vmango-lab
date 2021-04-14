import xsimlab as xs
from xsimlab.variable import VarIntent
import pandas as pd
import numpy as np
import igraph as ig
import io
import toml
import warnings
import pathlib
import IPython
import pgljupyter


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_inputs_from_df(model, df, cycle):
    topology_prc = model['topology']
    inputs = {}
    required_attrs = [
        'id',
        'parent_id',
        'cycle',
        'is_apical',
        'appearance_month',
        'ancestor_nature',
        'ancestor_is_apical',
        'nature'
    ]
    # all optinal attrs are related to arch dev
    optional_attrs = list(set([
        'burst_date',
        'flowering_date',
        'has_apical_child',
        'nb_lateral_children',
        'nb_inflo',
        'nb_fruit'
    ]) & set(df.columns))

    assert len(set(df.columns) & set(required_attrs)) == len(required_attrs)

    df = df[df['cycle'] <= cycle]
    edges = df[['id', 'parent_id']][pd.notna(df['parent_id'])]
    vertices = df[required_attrs + optional_attrs]

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
            inputs[f'topology__{var_name}'] = np.array(graph.vs.get_attribute_values(var_name), dtype=np.float32)
    inputs['topology__adjacency'] = np.array(graph.get_adjacency().data, dtype=np.float32)
    if 'arch_dev_rep' in model:
        inputs['arch_dev_rep__nature'] = np.array(graph.vs.get_attribute_values('nature'), dtype=np.float32)
    if 'arch_dev' in model:
        inputs['arch_dev__pot_nature'] = np.array(graph.vs.get_attribute_values('nature'), dtype=np.float32)
        for attr in optional_attrs:
            if attr == 'burst_date' or attr == 'flowering_date':
                inputs[f'arch_dev__pot_{attr}'] = np.array(graph.vs.get_attribute_values(attr), dtype='datetime64[D]')
            else:
                inputs[f'arch_dev__pot_{attr}'] = np.array(graph.vs.get_attribute_values(attr), dtype=np.float32)

    return inputs, graph


def create_setup(
    model,
    start_date,
    end_date,
    current_cycle,
    clocks={},
    tree=None,
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
            if 'initial_tree' not in setup and tree is None:
                raise ValueError('No initial tree provided')
            elif 'initial_tree' in setup:
                path = dir_path.joinpath(setup['initial_tree'])
                tree = pd.read_csv(path).astype(np.float32)

    graph = None
    if 'topology' in model:
        inputs, graph = get_inputs_from_df(model, tree, current_cycle)
        input_vars.update(inputs)
        input_vars['topology__current_cycle'] = current_cycle
        # work-around for main_clock not available at initialization.
        # set the start date variable
        input_vars['topology__sim_start_date'] = start_date

    output_vars_ = {}
    if type(output_vars) == dict:
        for name, item in output_vars.items():
            if type(item) == dict:
                for var_name, clock in item.items():
                    output_vars_[f'{name}__{var_name}'] = clock
            else:
                output_vars_[name] = item

    for prc_name in model:
        for var_name in xs.filter_variables(model[prc_name], var_type='variable', func=lambda var: var.metadata['intent'] == VarIntent.INOUT and 'GU' in list(sum(var.metadata['dims'], ()))):
            if type(output_vars) == dict:  # must be exported because of growing index
                if f'{prc_name}__{var_name}' not in output_vars:
                    output_vars[f'{prc_name}__{var_name}'] = None

    for prc_name in model:
        prc = model[prc_name]
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: var.metadata['static']):
            if f'{prc_name}__{var_name}' not in output_vars_:
                output_vars_[f'{prc_name}__{var_name}'] = None
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: not var.metadata['static']):
            if f'{prc_name}__{var_name}' not in output_vars_:
                output_vars_[f'{prc_name}__{var_name}'] = output_vars if type(output_vars) is str else None  # str must be clock name
        if graph is not None:
            # make simlab happy by passing initial 'inout' values (needlessly)
            shape = (len(graph.vs.indices),)
            for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: var.metadata['intent'] == VarIntent.INOUT and 'GU' in list(sum(var.metadata['dims'], ()))):
                if f'{prc_name}__{var_name}' not in input_vars:
                    input_vars[f'{prc_name}__{var_name}'] = np.full(shape, np.nan, dtype=np.float32)

    return xs.create_setup(
        model,
        clocks,
        main_clock,
        input_vars,
        output_vars_,
        fill_default
    )


def run(dataset, model, progress=True, geometry=False, hooks=[]):
    hooks = [xs.monitoring.ProgressBar()] + hooks if progress else hooks
    if geometry:
        sw = pgljupyter.SceneWidget(size_world=2.5)
        IPython.display.display(sw)

        @xs.runtime_hook(stage='run_step')
        def hook(model, context, state):
            scene = state[('geometry', 'scene')]
            if scene != sw.scenes[0]['scene']:
                sw.set_scenes(scene, scales=[1/100])

        hooks.append(hook)

    return dataset.xsimlab.run(model=model, decoding={'mask_and_scale': False}, hooks=hooks)
