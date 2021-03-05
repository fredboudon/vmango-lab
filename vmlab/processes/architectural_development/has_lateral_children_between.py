import xsimlab as xs
import numpy as np

from . import topology, has_veg_children_between, has_apical_child_between
from vmlab.processes import BaseProbabilityTableProcess


@xs.process
class HasLateralChildrenBetween(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    has_lateral_children_between = xs.variable(dims='GU', intent='out')

    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')
    seed = xs.foreign(topology.Topology, 'seed')
    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')

    nature = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'nature')
    has_veg_children_between = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'has_veg_children_between')
    has_apical_child_between = xs.foreign(has_apical_child_between.HasApicalChildBetween, 'has_apical_child_between')

    def initialize(self):
        self.has_lateral_children_between = np.array([])
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.has_lateral_children_between[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                self.has_lateral_children_between[
                    (self.has_veg_children_between == 1.) & (self.appeared == 1.) & (self.has_apical_child_between == 0.)
                ] = 1.
                for gu in np.flatnonzero((self.has_veg_children_between == 1.) & (self.appeared == 1.) & (self.has_apical_child_between == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values
                    if len(probability):
                        self.has_lateral_children_between[gu] = self.rng.binomial(1, probability[0])
