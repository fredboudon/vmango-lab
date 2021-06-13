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
from importlib import resources

pgl.pglParserVerbose(False)


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_vars_from_model(model, process_filter=None):
    """Convenience function to create output_vars dicts for vmlab.create_setup

    Returns a dict with 'foo__bar' attr for all non static variables
    of all processes or if there name is present in process_filter only
    of those specified.
    """

    var_names = []
    for prc_name in model:
        if process_filter is None or prc_name in process_filter:
            prc = model[prc_name]
            for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: not var.metadata['static']):
                var_names.append(f'{prc_name}__{var_name}')
    return var_names


def detect_processes(processpath='vmlab.processes'):
    """Find all processes available
    """
    import inspect
    import pkgutil
    import importlib
    base = importlib.import_module(processpath)

    result = {}
    found_packages = list(pkgutil.iter_modules(base.__path__))
    for mfinder, modname, ispkg in found_packages:
        if not modname.startswith('_'):
            submodule = importlib.import_module(processpath+'.'+modname)
            for objname in dir(submodule):
                obj = submodule.__dict__[objname]
                if inspect.isclass(obj):
                    result[objname] = obj
    return result


def _model_parameters(model):
    process_names = list(model.all_vars_dict.keys())
    processes = {}
    for p in process_names:
        processes[p] = model[p].__class__.__name__
    return processes


def _model_from_parameters(parameters):
    processes = detect_processes()
    arguments = dict([(name, processes[processname]) for name, processname in parameters.items()])
    return xs.Model(arguments)


def copy_model(model):
    """Create a copy of a vmlab model
    """
    return _model_from_parameters(_model_parameters(model))


def check_graph(graph):
    """Test if an igraph graph's structure is valid and usable in vmlab:

        - not empty
        - is directed and acyclic
        - all vertices are connected
        - at most one apical child per parent

    Parameters
    ----------
    graph : :class:`igraph.Graph` object
    """

    # tree must not be empty
    assert len(graph.vs.indices) > 0
    # tree must be directed and acyclic
    assert graph.is_dag()
    # all vertices must be connected (only one tree/component)
    assert len(graph.components('weak').sizes()) == 1
    # all vertices may have at most one apical child
    if 'topology__is_apical' in graph.vs.attribute_names():
        for v in graph.vs:
            successors = v.successors()
            nb_apical_children = 0
            for successor in successors:
                nb_apical_children = nb_apical_children + successor['topology__is_apical']
                assert nb_apical_children <= 1


def to_graph(df):
    """Load and validate an igraph graph from a pandas DataFrame

    Parameters
    ----------
    df : :class:`pandas.DataFrame` object
        Required format is identical with what is specified for the
        'tree' input in the vmlab.create_setup function.

    Returns
    -------
    graph : :class:`igraph.Graph`
    """

    assert 'id' in df.columns.to_list() and 'parent_id' in df.columns.to_list()

    edges = df[['parent_id', 'id']].dropna()
    vertices = df.drop('parent_id', axis=1) if len(df.columns.to_list()) > 2 else None
    graph = ig.Graph.DataFrame(edges, vertices=vertices)

    check_graph(graph)

    return graph


def to_dataframe(graph):
    """Load and validate an pandas DataFrame from an igraph graph

    Parameters
    ----------
    graph : :class:`igraph.Graph`

    Returns
    -------
    df : :class:`pandas.DataFrame` object
    """

    edges = np.argwhere(np.array(graph.get_adjacency().data))
    df = pd.DataFrame({
        'id': np.append(0, edges[:, 1]),
        'parent_id': np.append(np.nan, edges[:, 0])
    }).join(graph.get_vertex_dataframe())

    return df


def _get_inputs_from_graph(graph, model, cycle):

    all_vars_dict = model.all_vars_dict
    inputs = {}

    # drop all vertices where attr cycle > cycle if attr cycle is provided
    if 'topology__cycle' in graph.vs.attribute_names():
        graph.delete_vertices(graph.vs.select(topology__cycle_gt=cycle))
        check_graph(graph)
    else:
        graph.vs.set_attribute_values('topology__cycle', cycle)

    inputs['topology__adjacency'] = np.array(graph.get_adjacency().data, dtype=np.float32)

    for attr in graph.vs.attribute_names():
        if attr.find('__', 1, -1) > 0:
            prc_name, var_name = attr.split('__', maxsplit=1)
            if prc_name in all_vars_dict and var_name in all_vars_dict[prc_name]:
                # not sure what the best way is to figure out the date type
                inputs[attr] = np.array(graph.vs.get_attribute_values(attr)).astype('datetime64[ns]' if 'date' in var_name else np.float32)

    return inputs


def create_setup(
    model,
    start_date,
    end_date,
    setup_toml,
    current_cycle=3,
    tree=None,
    input_vars=None,
    output_vars=None
):
    """Create a vmlab setup for model runs

    Wraps the xarray-simlab (v0.5.0) create_setup function

    Parameters
    ----------
    model : :class:`xsimlab.Model` object
        Create a simulation setup for this model.
    start_date : str
        An iso date string (YYYY-MM-DD)
        Required to determine number of simulation steps (days).
    end_date : str
        An iso date string (YYYY-MM-DD).
        Required to determine number of simulation steps (days).
    setup_toml : str
        A custom file path or file name of the toml file that containes relevant
        parameters like paths to weather data, process parameters etc.
        If the file is not present it will look for the file name in
        the vmlab/data/setup folder.
    current_cycle : int, optional
        The number of the current (at start_date) vegetative cycle.
        If the cycle for each GU is provided in the input tree then
        all GUs of cycle > current_cycle will be automatically pruned
        from the initial tree. If they are not provided the cycle
        property of all initially present GUs will be set to 'current_cycle'
    tree : :class:`pandas.DataFrame object, optional
        A pandas DataFrame with at least the following columns:
        'id', 'parent_id' and 'topology__is_apical'
        Optionally any other input variable with a 'GU' dimension may be added.
        Column names follow the 'foo__bar' naming logic (see input_vars).
        If None it is expected that a path to a csv file is present in the
        setup_toml file.
    input_vars : dict, optional
        Dictionary with values given for model inputs. Entries of the
        dictionary may look like:
        - ``'foo': {'bar': value, ...}`` or
        - ``'foo__bar': value``
        where ``foo`` is the name of a existing process in the model and
        ``bar`` is the name of an (input) variable declared in that process.
        (copied from the xarray-simlab documentation)
    output_vars : dict, optional
        Dictionary with model variable names to save as simulation output
        (time-dependent or time-independent). Entries of the dictionary look
        similar than for ``input_vars`` (see here above) except: 'value' may
        be 'day' (daily output) or None (once at the end of the simulation)

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        A new Dataset object with model inputs as data variables or coordinates
        (depending on their given value) and clock coordinates.
        The names of the input variables also include the name of their process
        (i.e., 'foo__bar').
        (copied from the xarray-simlab documentation)
    """

    assert type(input_vars) == dict or input_vars is None
    assert type(output_vars) == dict or output_vars is None

    input_vars = {
        **({} if input_vars is None else input_vars)
    }

    main_clock = 'day'
    clocks = {}
    clocks[main_clock] = pd.date_range(start=start_date, end=end_date, freq='1d')

    # set toml file path as process input from 'parameters' section in setup_toml
    setup_toml_path = ''
    if pathlib.Path(setup_toml).is_file():
        setup_toml_path = setup_toml
    else:
        with resources.path('vmlab.data.setup', setup_toml) as path:
            setup_toml_path = path
    with io.open(setup_toml_path) as setup_file:
        setup = toml.loads(setup_file.read())
        dir_path = pathlib.Path(setup_toml_path).parent
        if 'parameters' in setup:
            for prc_name, rel_file_path in setup['parameters'].items():
                path = dir_path.joinpath(rel_file_path)
                if prc_name in model:
                    if path.exists():
                        # process 'prc_name' must inherit from ParameterizedProcess or
                        # declare a parameter_file_path 'in' variable and handle it
                        if f'{prc_name}__parameter_file_path' not in input_vars:
                            input_vars[f'{prc_name}__parameter_file_path'] = str(path)
                    else:
                        warnings.warn(f'Input file "{path}" does not exist')
        if tree is None:
            if 'initial_tree' in setup:
                path = dir_path.joinpath(setup['initial_tree'])
                tree = pd.read_csv(path)
            else:
                raise ValueError('No initial tree provided')

    graph = to_graph(tree)
    input_vars.update(_get_inputs_from_graph(graph, model, current_cycle))
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
                    if 'date' in var_name:
                        input_vars[f'{prc_name}__{var_name}'] = np.full(shape, np.datetime64('NaT'), dtype='datetime64[ns]')
                    else:
                        input_vars[f'{prc_name}__{var_name}'] = np.full(shape, np.nan, dtype=np.float32)

    return xs.create_setup(
        model,
        clocks,
        main_clock,
        input_vars,
        output_vars_,
    ).assign_attrs({
        # store as private attr so we can drop all outputs later that we just added to make zarr work with growing indices
        '__vmlab_output_vars': list(output_vars.keys()) if output_vars is not None else []
    })


def _cleaup_dataset(ds):

    # keep only those that were explicitly defined as output
    if '__vmlab_output_vars' in ds.attrs:
        ds = ds.drop_vars(set(ds.keys()).difference(ds.attrs['__vmlab_output_vars']))
    ds.attrs = {}

    # drop unused dims
    dims_to_drop = [k for k in ds.dims.keys()]
    for dim in ds.dims.keys():
        for data_var in ds.data_vars:
            if dim in ds[data_var].dims:
                dims_to_drop.remove(dim)
                break

    # it seems zarr creates some attrs and encoding settings that break writing to netcdf
    # https://github.com/pydata/xarray/issues/5223
    for data_var in ds.data_vars:
        ds[data_var].encoding = {}
        if '_FillValue' in ds[data_var].attrs:
            ds[data_var].attrs.pop('_FillValue')

    if len(dims_to_drop):
        ds = ds.drop_dims(dims_to_drop)

    return ds


def _fn_parallel(id, ds, geometry, store):

    if store is not None:
        store = f'{store}__{id}.zarr'

    @xs.runtime_hook(stage='finalize')
    def finalize(model, context, state):
        _fn_parallel.queue.put((id, 1))

    @xs.runtime_hook(stage='run_step')
    def run_step(model, context, state):
        _fn_parallel.queue.put((id, 0))
        if geometry:
            scene = state[('geometry', 'scene')]
            if scene is not None:
                _fn_parallel.queue.put((id, pgl.tobinarystring(scene, False)))

    hooks = [finalize, run_step]
    try:
        out = ds.xsimlab.run(_fn_parallel.model, decoding={'mask_and_scale': False}, hooks=hooks, store=store)
    except Exception:
        import traceback
        import logging
        logging.error(traceback.format_exc())
        _fn_parallel.queue.put((id, 1))
        return ds

    return out


def _f_init(queue, model_param):
    _fn_parallel.queue = queue
    _fn_parallel.model = _model_from_parameters(model_param)


def _run_parallel(ds, model, store, batch, sw, scenes, positions, progress, nb_proc):
    geometry = sw is not None
    batch_dim, batch_runs = batch
    jobs = [(i, ds.xsimlab.update_vars(model, input_vars=input_vars), geometry, store) for i, input_vars in enumerate(batch_runs)]

    queue = mp.Manager().Queue()
    nb_workers = min(len(jobs), nb_proc or mp.cpu_count())
    pool = mp.Pool(nb_workers, _f_init, [queue, _model_parameters(model)])

    results = pool.starmap_async(_fn_parallel, jobs, error_callback=lambda err: print(err))
    pool.close()

    done = 0
    nb_steps = len(jobs) * ds.day.values.shape[0] - len(jobs)

    if progress:
        with tqdm(total=nb_steps, bar_format='{bar} {percentage:3.0f}%') as bar:
            while done < len(jobs):
                id, got = queue.get()
                if got == 0:
                    bar.update()
                elif got == 1:
                    done += 1
                elif geometry:
                    scenes[id] = pgl.frombinarystring(got)
                    sw.set_scenes(scenes, scales=1/100, positions=positions)
            bar.close()
    else:
        while done < len(jobs):
            id, got = queue.get()
            if got == 0:
                pass
            elif got == 1:
                done += 1
            elif geometry:
                scenes[id] = pgl.frombinarystring(got)
                sw.set_scenes(scenes, scales=1/100, positions=positions)

    out = [_cleaup_dataset(ds) for ds in results.get()]
    pool.terminate()

    # if there is only one variable in the batch we set it as index with coords in concat
    if len(batch_runs) and all(x.keys() == batch_runs[0].keys() and len(x.keys()) == 1 for x in batch_runs):
        dim = xr.IndexVariable(batch_dim, [x[list(batch_runs[0].keys())[0]] for x in batch_runs])
    else:
        dim = batch_dim

    return xr.concat(out, dim=dim)


def run(dataset, model, progress=True, geometry=False, batch=None, store=None, hooks=[], nb_proc=None, verbosity=0):
    """Run a vmlab model

    Wraps the xarray-simlab (v0.5.0) run function

    Parameters
    ----------
    dataset : :class:`xarray.Dataset` object
        The dataset created with vmlab.create_setup
    model : :class:`xsimlab.Model` object
        A vmlab model
    progress : boolen, optional
        If true displays a progress bar
    geometry : dict or boolean, optional
        If true or dict: displays a pgljupyter SceneWidget.
        The dict may have the properties
            - size : int, size [m] of a cell in batch mode or size [m] of plane in normal model
            - size_display: tuple with width and height of the canvas
    batch : tuple, optional
        A tuple of length 2 with a name and an array of dicts (input_vars)
    store : str, optional
        A file path. In batch mode each batch's path will be extended with its index
    hooks : list, optional
        One or more xarray-simlab runtime hooks
    nb_proc : int, optional
        Maximum number of processes created in batch muti-process processing.
        Defaults to minimum(number of CPU cores available, number of batches)
    verbosity : int, optional
        0 = no vmlab warnings, 1 = all warnings

    Returns
    -------
    output : :class:`xarray.Dataset` object
        A Datset with those variables that have been defined in 'output_vars'.
        All dimensions not used in any variable will be droped automatically.
        In batch mode if there is only one variable in each batch the resulting
        Datasets will be concatenanted using the batch name as index name
        and its values as coordinates.
    """

    hooks = [xs.monitoring.ProgressBar()] + hooks if progress else hooks
    is_batch_run = type(batch) == tuple
    sw = None
    scenes = []
    positions = []
    size = 2.5
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
                if scene is not None:
                    sw.set_scenes(scene, scales=1/100)
            hooks.append(hook)

        sw = pgljupyter.SceneWidget(size_world=size, size_display=size_display)
        IPython.display.display(sw)

        if verbosity == 0:
            warnings.filterwarnings('ignore', 'vmlab*')
        else:
            warnings.filterwarnings('default', 'vmlab*')

        if is_batch_run:
            with model:
                ds = _run_parallel(dataset, model, store, batch, sw, scenes, positions, progress, nb_proc)
        else:
            ds = dataset.xsimlab.run(model=model, decoding={'mask_and_scale': False}, hooks=hooks, store=store)

    return _cleaup_dataset(ds)
