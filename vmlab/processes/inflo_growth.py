import xsimlab as xs
import numpy as np

from . import parameters
from . import growth_unit_growth
from .base import BaseGrowthUnitProcess

@xs.process
class InfloGrowth(BaseGrowthUnitProcess):

    params = xs.foreign(parameters.Parameters, 'inflo_growth')

    GU = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'GU')

    def initialize(self):
        self.DAB = np.ones(self.GU.shape) * -1

    def step(self, nsteps, step, step_start, step_end, step_delta):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
