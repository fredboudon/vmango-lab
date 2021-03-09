import xsimlab as xs
import numpy as np

from . import topology, fruiting, nb_inflorescences
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class NbFruits(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    nb_fruits = xs.variable(dims='GU', intent='out')

    GU = xs.foreign(topology.Topology, 'GU')
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
    nb_inflorescences = xs.foreign(nb_inflorescences.NbInflorescences, 'nb_inflorescences')

    def initialize(self):
        self.nb_fruits = np.zeros(self.GU.shape)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.nb_fruits[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                if np.any((self.appeared == 1.) & (self.fruiting == 1.)):
                    gu_indices = np.nonzero((self.appeared == 1.) & (self.fruiting == 1.))
                    indices = self.get_indices(tbl, gu_indices)
                    probability = tbl.loc[indices.tolist()].values.flatten()
                    self.nb_fruits[gu_indices] = self.rng.binomial(1, probability, probability.shape).astype(np.int) + 1
