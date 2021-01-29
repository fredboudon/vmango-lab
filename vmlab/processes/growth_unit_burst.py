import xsimlab as xs
import numpy as np
import sparse

from . import topology
from . import phenology
from .base import BaseGrowthUnitProcess


@xs.process
class GrowthUnitBurst(BaseGrowthUnitProcess):

    is_apical = xs.foreign(topology.Topology, 'is_apical')
    adjacency = xs.foreign(topology.Topology, 'adjacency')

    gu_growth_tts = xs.foreign(phenology.Phenology, 'gu_growth_tts')

    gu_bursted = xs.variable(
        dims=('GU'),
        intent='inout',
        groups=('bursts')
    )

    gu_bursted_is_apical = xs.variable(
        dims=('GU'),
        intent='inout',
        groups=('bursts')
    )

    def initialize(self):

        self.gu_bursted = np.array(self.gu_bursted, dtype=np.bool)
        self.gu_bursted_is_apical = np.array(self.gu_bursted_is_apical, dtype=np.bool)

    def step(self, nsteps, step, step_start, step_end, step_delta):

        # burst happens at different temperature sums depending on is_apical
        self.gu_bursted = ((self.gu_growth_tts >= 300) & self.is_apical) | ((self.gu_growth_tts >= 400) & ~self.is_apical)
        if np.any(self.gu_bursted):
            # find all parents and set the GU index to True if the parent is apical
            # sparse should in theory we faster. Remains to be tested.
            # print(self.adjacency.shape, self.gu_bursted.shape, self.is_apical.shape)
            parent_is_apical_sparse = sparse.COO.from_numpy(self.adjacency) * self.gu_bursted * self.is_apical
            parent_is_apical = np.bitwise_or.reduce(parent_is_apical_sparse.todense())
            probability_of_child_being_apical_if_parent_is_apical = 0.4
            self.gu_bursted_is_apical = parent_is_apical & np.random.choice((1,0), len(self.gu_bursted), probability_of_child_being_apical_if_parent_is_apical) + \
                ~parent_is_apical & np.random.choice((1,0), len(self.gu_bursted), 1 - probability_of_child_being_apical_if_parent_is_apical)

    def finalize_step(self):
        pass

    def finalize(self):
        pass
