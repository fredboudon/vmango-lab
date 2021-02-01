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

    gu_stage = xs.foreign(phenology.GrowthUnitPhenology, 'gu_stage')
    gu_growth_tts = xs.foreign(phenology.GrowthUnitPhenology, 'gu_growth_tts')

    burst_t_apical = xs.variable(
        dims=(),
        static=True
    )

    burst_t_lateral = xs.variable(
        dims=(),
        static=True
    )

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

        # burst happens at different temperature sums at stage=D depending on is_apical
        # has children
        has_children = np.bitwise_or.reduce(self.adjacency.transpose())
        stage_D = self.gu_stage == 'D'
        self.gu_bursted = stage_D & np.random.choice((True, False), self.adjacency.shape[0]) & (((self.gu_growth_tts >= self.burst_t_apical) & self.is_apical) | ((self.gu_growth_tts >= self.burst_t_lateral) & ~self.is_apical))
        if np.any(self.gu_bursted):
            self.gu_bursted_is_apical = self.gu_bursted & ~has_children

    def finalize_step(self):
        pass

    def finalize(self):
        pass
