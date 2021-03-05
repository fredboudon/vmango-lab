import xsimlab as xs
import numpy as np

from . import topology, flowering, nb_inflorescences
from vmlab.processes import BaseProbabilityTableProcess


@xs.process
class Fruiting(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    fruiting = xs.variable(dims='GU', intent='out')

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
    nb_inflorescences = xs.foreign(nb_inflorescences.NbInflorescences, 'nb_inflorescences')

    def initialize(self):
        self.fruiting = np.array([])
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.fruiting[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.flowering == 1.) & (self.appeared == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values
                    if len(probability):
                        self.fruiting[gu] = self.rng.binomial(1, probability[0])
