import xsimlab as xs
import numpy as np

from . import topology, has_veg_children_within, has_mixed_inflo_children_between
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class Flowering(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    flowering = xs.variable(dims='GU', intent='out')

    GU = xs.foreign(topology.Topology, 'GU')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')

    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')
    has_mixed_inflo_children_between = xs.foreign(has_mixed_inflo_children_between.HasMixedInfloChildrenBetween, 'has_mixed_inflo_children_between')

    def initialize(self):
        self.flowering = np.zeros(self.GU.shape)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):

            self.flowering[self.appeared == 1.] = 0.

            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                if np.any((self.appeared == 1.) & (self.has_veg_children_within == 0.) & (self.has_mixed_inflo_children_between == 0.)):
                    gu_indices = np.nonzero((self.appeared == 1.) & (self.has_veg_children_within == 0.) & (self.has_mixed_inflo_children_between == 0.))
                    indices = self.get_indices(tbl, gu_indices)
                    probability = tbl.loc[indices.tolist()].values.flatten()
                    self.flowering[gu_indices] = self.rng.binomial(1, probability, probability.shape)
