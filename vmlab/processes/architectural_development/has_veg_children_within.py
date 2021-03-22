import xsimlab as xs
import numpy as np

from . import topology
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class HasVegChildrenWithin(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    has_veg_children_within = xs.variable(dims='GU', intent='out')

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
    ancestor_is_apical = xs.foreign(topology.Topology, 'ancestor_is_apical')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    def initialize(self):
        self.has_veg_children_within = np.zeros(self.GU.shape)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step'))
    def run_step(self, step):
        if np.any(self.appeared):
            self.has_veg_children_within[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                gu_indices = np.nonzero(self.appeared)
                indices = self.get_indices(tbl, gu_indices)
                probability = tbl.loc[indices.tolist()].values.flatten()
                self.has_veg_children_within[gu_indices] = self.rng.binomial(1, probability, probability.shape)
