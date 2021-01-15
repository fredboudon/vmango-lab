import xsimlab as xs

from . import parameters
from . import architecture


@xs.process
class GrowthUnitGrowth():

    params = xs.foreign(parameters.Parameters, 'growth_unit_growth')
    GU = xs.foreign(architecture.Architecture, 'GU')

    def initialize(self):
        pass

    @xs.runtime(args=('step'))
    def run_step(self, step):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
