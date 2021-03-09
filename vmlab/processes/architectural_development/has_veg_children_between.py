import xsimlab as xs
import numpy as np

from . import topology, has_veg_children_within, fruiting, flowering
from ._base.probability_table import BaseProbabilityTableProcess
from ...enums import Nature


@xs.process
class HasVegChildrenBetween(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    has_veg_children_between = xs.variable(dims='GU', intent='out')
    nature = xs.variable(dims='GU', intent='out')

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

    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')
    fruiting = xs.foreign(fruiting.Fruiting, 'fruiting')
    flowering = xs.foreign(flowering.Flowering, 'flowering')

    def initialize(self):
        self.has_veg_children_between = np.zeros(self.GU.shape)
        self.nature = np.full(self.GU.shape, Nature.VEGETATIVE)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step'))
    def run_step(self, step):
        if np.any(self.appeared):

            appeared = self.appeared == 1.
            fruiting = self.fruiting == 1.
            flowering = self.flowering == 1.
            not_has_veg_children_within = self.has_veg_children_within == 0.

            self.has_veg_children_between[appeared] = 0.

            self.nature[appeared] = Nature.VEGETATIVE
            self.nature[appeared & fruiting] = Nature.FRUITING
            self.nature[appeared & ~fruiting & flowering] = Nature.PURE_FLOWER

            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                if np.any(not_has_veg_children_within):
                    gu_indices = np.nonzero(appeared & not_has_veg_children_within)
                    indices = self.get_indices(tbl, gu_indices)
                    probability = tbl.loc[indices.tolist()].values.flatten()
                    self.has_veg_children_between[gu_indices] = self.rng.binomial(1, probability, probability.shape)
