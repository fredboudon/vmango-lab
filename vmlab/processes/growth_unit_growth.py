import xsimlab as xs
import numpy as np

from . import parameters


@xs.process
class GrowthUnitGrowth():

    GU = xs.index(dims=('GU'), groups=('topology'))

    params = xs.foreign(parameters.Parameters, 'growth_unit_growth')

    nb_leaves = xs.variable(
        dims=('GU'),
        intent='inout'
    )

    def initialize(self):

        self.GU = np.arange(0, len(self.nb_leaves), step=1, dtype=np.int64)
        self.nb_leaves = np.array(self.nb_leaves, dtype=np.int64)

    @xs.runtime(args=('step'))
    def run_step(self, step):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
