import xsimlab as xs
import numpy as np

from . import topology, fruiting
from ._base import BaseProbabilityTable


@xs.process
class NbFruits(BaseProbabilityTable):

    rng = xs.global_ref('rng')

    path = xs.variable()
    probability_tables = xs.any_object()

    nb_fruits = xs.variable(dims='GU', intent='out')

    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')
    seed = xs.foreign(topology.Topology, 'seed')
    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    fruiting = xs.foreign(fruiting.Fruiting, 'fruiting')

    def initialize(self):
        self.nb_fruits = np.array([])
        self.probability_tables = self.get_probability_tables(self.path)

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.fruiting == 1.) & (self.appeared == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values
                    if len(probability):
                        self.nb_fruits[gu] = int(self.rng.binomial(1, probability[0])) + 1
