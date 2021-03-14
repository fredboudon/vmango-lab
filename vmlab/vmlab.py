import xsimlab as xs
import io
import toml
import warnings
import pathlib


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_setup(
    model=None,
    clocks=None,
    main_clock=None,
    input_vars=None,
    output_vars=None,
    fill_default=True,
    setup_toml=None
):
    """Create a specific setup for model runs.

    This convenient function creates a new :class:`xarray.Dataset`
    object with everything needed to run a model (i.e., input values,
    time steps, output variables to save at given times) as data
    variables, coordinates and attributes.

    Parameters
    ----------
    model : :class:`xsimlab.Model` object, optional
        Create a simulation setup for this model. If None, tries to get model
        from context.
    clocks : dict, optional
        Used to create one or several clock coordinates. Dictionary
        values are anything that can be easily converted to
        :class:`xarray.IndexVariable` objects (e.g., a 1-d
        :class:`numpy.ndarray` or a :class:`pandas.Index`).
    main_clock : str or dict, optional
        Name of the clock coordinate (dimension) to use as master clock.
        If not set, the name is inferred from ``clocks`` (only if
        one coordinate is given and if Dataset has no master clock
        defined yet).
        A dictionary can also be given with one of several of these keys:

        - ``dim`` : name of the master clock dimension/coordinate
        - ``units`` : units of all clock coordinate labels
        - ``calendar`` : a unique calendar for all (time) clock coordinates
    input_vars : dict, optional
        Dictionary with values given for model inputs. Entries of the
        dictionary may look like:

        - ``'foo': {'bar': value, ...}`` or
        - ``('foo', 'bar'): value`` or
        - ``'foo__bar': value``

        where ``foo`` is the name of a existing process in the model and
        ``bar`` is the name of an (input) variable declared in that process.

        Values are anything that can be easily converted to
        :class:`xarray.Variable` objects, e.g., single values, array-like,
        ``(dims, data, attrs)`` tuples or xarray objects.
        For array-like values with no dimension labels, xarray-simlab will look
        in ``model`` variables metadata for labels matching the number
        of dimensions of those arrays.
    output_vars : dict, optional
        Dictionary with model variable names to save as simulation output
        (time-dependent or time-independent). Entries of the dictionary look
        similar than for ``input_vars`` (see here above), except that here
        ``value`` must correspond to the dimension of a clock coordinate
        (i.e., new output values will be saved at each time given by the
        coordinate labels) or ``None`` (i.e., all variables will be exported).
    fill_default : bool, optional
        If True (default), automatically fill the dataset with all model
        inputs missing in ``input_vars`` and their default value (if any).

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        A new Dataset object with model inputs as data variables or coordinates
        (depending on their given value) and clock coordinates.
        The names of the input variables also include the name of their process
        (i.e., 'foo__bar').

    Notes
    -----
    Output variable names are added in Dataset as specific attributes
    (global and/or clock coordinate attributes).

    Shamelessly copied from xs.create_setup. But we export all if output_vars=None
    """

    input_vars = {
        **({} if input_vars is None else input_vars)
    }

    if main_clock is None and len(clocks.keys()):
        main_clock = list(clocks.keys())[0]

    # set toml file path as process input from 'parameters' section in setup_toml
    if setup_toml is not None:
        with io.open(setup_toml) as setup_file:
            setup = toml.loads(setup_file.read())
            if 'parameters' in setup:
                dir_path = pathlib.Path(setup_toml).parent
                for prc_name, rel_file_path in setup['parameters'].items():
                    path = dir_path.joinpath(rel_file_path)
                    if not path.exists():
                        warnings.warn(f'Input file "{path}" does not exist')
                    elif prc_name in model:
                        # process 'prc_name' must inherit from BaseParameterizedProcess or
                        # declare a parameter_file_path 'in' variable and handle it
                        input_vars[f'{prc_name}__parameter_file_path'] = str(path)
                    else:
                        warnings.warn(f'Process "{prc_name}" does not exist')

    # set toml file path as process input from 'probability_tables' section in setup_toml
    if setup_toml is not None:
        with io.open(setup_toml) as setup_file:
            setup = toml.loads(setup_file.read())
            if 'probability_tables' in setup:
                dir_path = pathlib.Path(setup_toml).parent
                for prc_name, rel_dir_path in setup['probability_tables'].items():
                    path = dir_path.joinpath(rel_dir_path)
                    if not path.exists():
                        warnings.warn(f'Input dir "{path}" does not exist')
                    elif prc_name in model:
                        # process 'prc_name' must inherit from BaseProbabilityTableProcess
                        input_vars[f'{prc_name}__table_dir_path'] = str(path)
                    else:
                        warnings.warn(f'Process "{prc_name}" does not exist')

    output_vars_ = {}
    for prc_name in model:
        output_vars_[prc_name] = {}
        prc = model[prc_name]
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: var.metadata['static']):
            output_vars_[prc_name][var_name] = None
        for var_name in xs.filter_variables(prc, var_type='variable', func=lambda var: not var.metadata['static']):
            output_vars_[prc_name][var_name] = output_vars if type(output_vars) is str else None  # str must be clock name

    return xs.create_setup(
        model,
        clocks,
        main_clock,
        input_vars,
        output_vars_,
        fill_default
    )
