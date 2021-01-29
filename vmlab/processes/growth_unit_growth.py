import xsimlab as xs
# import numpy as np

from . import parameters
from . import topology
from . import phenology
from .base import BaseGrowthUnitProcess


@xs.process
class GrowthUnitGrowth(BaseGrowthUnitProcess):

    params = xs.foreign(parameters.Parameters, 'growth_unit_growth')

    GU = xs.foreign(topology.Topology, 'GU')

    gu_growth_tts = xs.foreign(phenology.Phenology, 'gu_growth_tts')
    leaf_growth_tts = xs.foreign(phenology.Phenology, 'leaf_growth_tts')
    gu_pheno_tts = xs.foreign(phenology.Phenology, 'gu_pheno_tts')

    def initialize(self):
        pass

    def step(self, nsteps, step, step_start, step_end, step_delta):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
