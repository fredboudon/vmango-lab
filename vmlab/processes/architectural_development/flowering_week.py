import xsimlab as xs
import numpy as np

from . import topology, flowering
from vmlab.processes import BaseProbabilityTableProcess


@xs.process
class FloweringWeek(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    flowering_week = xs.variable(dims='GU', intent='out')

    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')
    seed = xs.foreign(topology.Topology, 'seed')
    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')

    flowering = xs.foreign(flowering.Flowering, 'flowering')

    def initialize(self):
        self.flowering_week = np.array([])
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.flowering == 1.) & (self.appeared == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probabilities = tbl[tbl.index == index].values.flatten()
                    if len(probabilities):
                        realization = np.flatnonzero(self.rng.multinomial(1, probabilities))[0]
                        week = int(tbl.columns[realization])
                        self.flowering_week[gu] = week
