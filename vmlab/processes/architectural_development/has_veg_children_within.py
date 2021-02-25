import xsimlab as xs
import numpy as np

from . import topology
from ._base import ProbabilityTableBase


@xs.process
class HasVegChildrenWithin(ProbabilityTableBase):

    rng = xs.global_ref('rng')

    path = xs.variable()
    probability_tables = xs.any_object()

    has_veg_children_within = xs.variable(dims='GU', intent='out')

    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')
    seed = xs.foreign(topology.Topology, 'seed')
    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    nb_fruits = xs.foreign(topology.Topology, 'nb_fruits')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    ancestor_is_apical = xs.foreign(topology.Topology, 'ancestor_is_apical')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    def initialize(self):
        self.has_veg_children_within = np.zeros(1)
        self.probability_tables = self.get_probability_tables(self.path)

    @xs.runtime(args=('step'))
    def run_step(self, step):
        if np.any(self.appeared):
            self.has_veg_children_within[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero(self.appeared):
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values
                    if len(probability):
                        self.has_veg_children_within[gu] = self.rng.binomial(1, probability[0])
