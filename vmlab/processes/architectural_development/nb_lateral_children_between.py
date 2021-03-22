import xsimlab as xs
import numpy as np

from . import topology, has_apical_child_between, has_lateral_children_between, has_veg_children_between
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class NbLateralChildrenBetween(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    nb_lateral_children_between = xs.variable(dims='GU', intent='out')

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

    nature = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'nature')
    has_lateral_children_between = xs.foreign(has_lateral_children_between.HasLateralChildrenBetween, 'has_lateral_children_between')
    has_apical_child_between = xs.foreign(has_apical_child_between.HasApicalChildBetween, 'has_apical_child_between')

    def initialize(self):
        self.nb_lateral_children_between = np.zeros(self.GU.shape)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.nb_lateral_children_between[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                gu_indices = np.nonzero(self.appeared)
                indices = self.get_indices(tbl, gu_indices)
                probability = tbl.loc[indices.tolist()].values.flatten()
                self.nb_lateral_children_between[gu_indices] = self.rng.binomial(1, probability, probability.shape) + 1
