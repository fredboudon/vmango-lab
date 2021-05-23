import io
import toml
import numpy as np
import xsimlab as xs
from xsimlab.variable import VarIntent

from vmlab import DotDict


@xs.process
class ParameterizedProcess:
    """Base process class to handle input parameter files.
    """

    parameter_file_path = xs.variable(intent='in', static=True, default=None)
    parameters = xs.any_object()

    # must be called by the inheriting class
    def initialize(self):
        if self.parameter_file_path is None:
            raise ValueError('Parameter file path not set')
        else:
            with io.open(self.parameter_file_path) as param_file:
                self.parameters = toml.loads(param_file.read(), _dict=DotDict)
                variables = xs.filter_variables(self.__xsimlab_cls__)
                # If the name of a parameter is also implemented as 'in' or 'inout' variable
                # and its current value is nan or its default value it is set form the parameter file
                # Priorities are
                # 1. value provided in 'input_vars' dict during setup
                # 2. value provided in parameter toml
                # 3. default value provided in xs.variable declaration
                for parameter in self.parameters:
                    if parameter in variables and variables[parameter].metadata['intent'] in (VarIntent.INOUT, VarIntent.IN):
                        value = getattr(self, parameter)
                        if value == variables[parameter].default or (type(value) == float and np.isnan(value)):
                            setattr(self, parameter, self.parameters[parameter])
