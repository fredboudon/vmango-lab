import xsimlab as xs
import numpy as np

from . import topology, has_veg_children_within, has_mixed_inflo_children_between
from vmlab.processes import BaseProbabilityTableProcess


@xs.process
class Flowering(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    flowering = xs.variable(dims='GU', intent='out')

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

    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')
    has_mixed_inflo_children_between = xs.foreign(has_mixed_inflo_children_between.HasMixedInfloChildrenBetween, 'has_mixed_inflo_children_between')

    def initialize(self):
        self.flowering = np.array([])
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.flowering[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.appeared == 1.) & (self.has_veg_children_within == 0.) & (self.has_mixed_inflo_children_between == 0.)):
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values
                    if len(probability):
                        self.flowering[gu] = self.rng.binomial(1, probability[0])
