import xsimlab as xs
import numpy as np

from . import topology
from . import phenology
from .base import BaseGrowthUnitProcess


@xs.process
class GrowthUnitNoBurst(BaseGrowthUnitProcess):

    GU = xs.foreign(topology.Topology, 'GU')

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

        self.gu_bursted = np.zeros(self.GU.shape, dtype=np.bool)
        self.gu_bursted_is_apical = np.zeros(self.GU.shape, dtype=np.bool)

    def step(self, nsteps, step, step_start, step_end, step_delta):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass


@xs.process
class GrowthUnitBurst(BaseGrowthUnitProcess):

    is_apical = xs.foreign(topology.Topology, 'is_apical')
    adjacency = xs.foreign(topology.Topology, 'adjacency')

    gu_growth_tts = xs.foreign(phenology.GrowthUnitPhenology, 'gu_growth_tts')

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
        # has children
        has_children = np.bitwise_or.reduce(self.adjacency.transpose())
        self.gu_bursted = np.random.choice((True, False), self.adjacency.shape[0]) & (((self.gu_growth_tts >= 150) & self.is_apical) | ((self.gu_growth_tts >= 300) & ~self.is_apical))  # & ~has_children
        if np.any(self.gu_bursted):
            self.gu_bursted_is_apical = self.is_apical * self.gu_bursted

    def finalize_step(self):
        pass

    def finalize(self):
        pass
