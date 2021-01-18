import io
import toml
import warnings
import pathlib
import numpy as np
import random

import xsimlab as xs


class DotDict(dict):

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@xs.process
class Parameters():

    path = xs.variable(intent='in', static=True)
    seed = xs.variable(intent='in', static=True)

    architecture = xs.any_object()
    carbon_balance = xs.any_object()
    environment = xs.any_object()
    fruit_growth = xs.any_object()
    inflo_growth = xs.any_object()
    growth_unit_growth = xs.any_object()
    fruit_quality = xs.any_object()
    light_interception = xs.any_object()
    photosynthesis = xs.any_object()

    def initialize(self):

        random.seed(self.seed)
        np.random.seed(self.seed)

        with io.open(self.path) as sim_file:
            settings = toml.loads(sim_file.read())
            for name, filename in settings['parameters'].items():
                try:
                    path = pathlib.Path(self.path).parent.joinpath(filename)
                    with io.open(path) as param_file:
                        setattr(self, name, (path, toml.loads(param_file.read(), _dict=DotDict)))
                except AttributeError:
                    warnings.warn(f'process "{name}" does not exist')
