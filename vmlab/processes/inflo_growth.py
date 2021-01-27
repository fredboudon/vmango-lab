import xsimlab as xs
import numpy as np

from . import parameters
from . import growth_unit_growth


@xs.process
class InfloGrowth():

    params = xs.foreign(parameters.Parameters, 'inflo_growth')

    GU = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'GU')

    def initialize(self):
        self.DAB = np.ones(self.GU.shape) * -1

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
