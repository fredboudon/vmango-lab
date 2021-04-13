import io
import toml
import xsimlab as xs

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
