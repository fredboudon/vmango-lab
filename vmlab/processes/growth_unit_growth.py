import xsimlab as xs
# import numpy as np

from . import parameters
from . import topology
from . import phenology


@xs.process
class GrowthUnitGrowth:

    params = xs.foreign(parameters.Parameters, 'growth_unit_growth')

    GU = xs.foreign(topology.Topology, 'GU')

    gu_growth_tts = xs.foreign(phenology.GrowthUnitPhenology, 'gu_growth_tts')

    def initialize(self):
        pass

    @xs.runtime(args=())
    def run_step(self):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
