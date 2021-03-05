import xsimlab as xs
import numpy as np

from . import topology, has_lateral_children_between, has_veg_children_between
from vmlab.processes import BaseProbabilityTableProcess


@xs.process
class NbLateralChildrenBetween(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    nb_lateral_children_between = xs.variable(dims='GU', intent='out')

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

    nature = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'nature')
    has_lateral_children_between = xs.foreign(has_lateral_children_between.HasLateralChildrenBetween, 'has_lateral_children_between')

    def initialize(self):
        self.nb_lateral_children_between = np.array([])
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.nb_lateral_children_between[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.has_lateral_children_between == 1.) & (self.appeared == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values[0]
                    probability = 1. if probability > 1. else probability
                    self.nb_lateral_children_between[gu] = self.rng.binomial(1, probability) + 1
