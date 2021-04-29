import io
import warnings
import multiprocessing as mp
import xsimlab as xs
import xarray as xr
from xsimlab.variable import VarIntent
import pandas as pd
import numpy as np
import igraph as ig
import openalea.plantgl.all as pgl
from tqdm.auto import tqdm
import toml
import pathlib
import IPython
import pgljupyter


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_vars_from_model(model, process_filter=None):
    var_names = []
    for prc_name in model:
        if process_filter is None or prc_name in process_filter:
            prc = model[prc_name]
            for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: not var.metadata['static']):
                var_names.append(f'{prc_name}__{var_name}')
    return var_names


def _get_inputs_from_df(model, df, cycle):
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
        'nb_fruit',
        'nb_internode',
        'final_length_gu'
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
            if attr in ['burst_date', 'flowering_date', 'has_apical_child', 'nb_lateral_children', 'nb_inflo', 'nb_fruit']:
                if attr == 'burst_date' or attr == 'flowering_date':
                    inputs[f'arch_dev__pot_{attr}'] = np.array(graph.vs.get_attribute_values(attr), dtype='datetime64[ns]')
                else:
                    inputs[f'arch_dev__pot_{attr}'] = np.array(graph.vs.get_attribute_values(attr), dtype=np.float32)
    if 'appearance' in model:
        for attr in optional_attrs:
            if attr in ['nb_internode', 'final_length_gu']:
                inputs[f'appearance__{attr}'] = np.array(graph.vs.get_attribute_values(attr), dtype=np.float32)

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
            if tree is None:
                if 'initial_tree' in setup:
                    path = dir_path.joinpath(setup['initial_tree'])
                    tree = pd.read_csv(path).astype(np.float32)
                else:
                    raise ValueError('No initial tree provided')

    graph = None
    if 'topology' in model:
        inputs, graph = _get_inputs_from_df(model, tree, current_cycle)
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
        output_vars = output_vars_.copy()

    for prc_name in model:
        prc = model[prc_name]
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: var.metadata['static']):
            if f'{prc_name}__{var_name}' not in output_vars_:
                output_vars_[f'{prc_name}__{var_name}'] = None
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: not var.metadata['static']):
            if f'{prc_name}__{var_name}' not in output_vars_:
                output_vars_[f'{prc_name}__{var_name}'] = output_vars if type(output_vars) is str else None  # str must be clock name
        if graph is not None:
            # make simlab happy by passing initial 'inout' values used to model cycles (needlessly)
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
    ).assign_attrs({
        '__vmlab_output_vars': list(output_vars.keys()) if output_vars is not None else []
    })


def _fn_parallel(id, ds, geometry):

    @xs.runtime_hook(stage='finalize')
    def finalize(model, context, state):
        _fn_parallel.q.put((id, 1))

    @xs.runtime_hook(stage='run_step')
    def run_step(model, context, state):
        _fn_parallel.q.put((id, 0))
        if geometry:
            scene_ = state[('geometry', 'scene')]
            if scene_ != run_step.scene:
                _fn_parallel.q.put((id, pgl.tobinarystring(scene_, False)))
                run_step.scene = scene_
    run_step.scene = None
    hooks = [finalize, run_step]
    try:
        out = ds.xsimlab.run(decoding={'mask_and_scale': False}, hooks=hooks)
    except Exception:
        import traceback
        import logging
        logging.error(traceback.format_exc())
        _fn_parallel.q.put((id, 1))
        return ds

    return out


def _run_parallel(ds, model, batch, sw, scenes, positions):
    # TODO: exceptions not catched in main thread

    geometry = sw is not None
    batch_dim, batch_runs = batch
    jobs = [(i, ds.xsimlab.update_vars(model, input_vars=input_vars), geometry) for i, input_vars in enumerate(batch_runs)]

    def f_init(q):
        _fn_parallel.q = q

    m = mp.Manager()
    q = m.Queue()
    nb_workers = max(1, mp.cpu_count() - 1)
    p = mp.Pool(nb_workers, f_init, [q])

    results = p.starmap_async(_fn_parallel, jobs, error_callback=lambda err: print(err))
    p.close()

    done = 0
    nb_steps = len(jobs) * ds.day.values.shape[0] - len(jobs)
    with tqdm(total=nb_steps, bar_format='{bar} {percentage:3.0f}%') as bar:
        while done < len(jobs):
            id, got = q.get()
            if got == 0:
                bar.update()
            elif got == 1:
                done += got
            else:
                scenes[id] = pgl.frombinarystring(got)
                sw.set_scenes(scenes, scales=1/100, positions=positions)
        bar.close()

    dss = results.get()
    p.terminate()

    return xr.concat(dss, dim=batch_dim)


def run(dataset, model, progress=True, geometry=False, hooks=[], batch=None):
    hooks = [xs.monitoring.ProgressBar()] + hooks if progress else hooks
    is_batch_run = type(batch) == tuple
    sw = None
    scenes = []
    positions = []
    size = 1
    size_display = (400, 400)
    if geometry:
        if type(geometry) == dict:
            size = size if 'size' not in geometry else geometry['size']
            size_display = size_display if 'size_display' not in geometry else geometry['size_display']
        if is_batch_run:
            from math import ceil, sqrt, floor
            length = len(batch[1])
            positions = []
            cell = size
            rows = cols = ceil(sqrt(length))
            size = rows * cell
            start = -size / 2 + cell / 2
            scenes = [None] * length
            for i in range(length):
                row = floor(i / rows)
                col = (i - row * cols)
                x = row * cell + start
                y = col * cell + start
                positions.append((x, y, 0))
        else:
            @xs.runtime_hook(stage='run_step')
            def hook(model, context, state):
                scene = state[('geometry', 'scene')]
                if scene != sw.scenes[0]['scene']:
                    sw.set_scenes(scene, scales=[1 / 100])
            hooks.append(hook)

        sw = pgljupyter.SceneWidget(size_world=size, size_display=size_display)
        IPython.display.display(sw)

    if is_batch_run:
        with model:
            ds = _run_parallel(dataset, model, batch, sw, scenes, positions)
    else:
        ds = dataset.xsimlab.run(model=model, decoding={'mask_and_scale': True}, hooks=hooks)

    ds = ds.drop_vars(set(ds.keys()).difference(ds.attrs['__vmlab_output_vars']))
    ds.attrs = {}

    dims_to_drop = [k for k in ds.dims.keys()]
    for dim in ds.dims.keys():
        for data_var in ds.data_vars:
            if dim in ds[data_var].dims:
                dims_to_drop.remove(dim)
                break
    for data_var in ds.data_vars:
        if '_FillValue' in ds[data_var].attrs:
            ds[data_var].attrs.pop('_FillValue')
    if len(dims_to_drop):
        ds = ds.drop_dims(dims_to_drop)

    return ds
