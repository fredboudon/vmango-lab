import xsimlab as xs
import numpy as np

from . import topology, has_veg_children_within, fruiting, flowering
from ._base import BaseProbabilityTable
from ...enums import Nature


@xs.process
class HasVegChildrenBetween(BaseProbabilityTable):

    rng = xs.global_ref('rng')

    path = xs.variable()
    probability_tables = xs.any_object()

    has_veg_children_between = xs.variable(dims='GU', intent='out')
    nature = xs.variable(dims='GU', intent='out')

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
        self.has_veg_children_between = np.zeros(1)
        self.nature = np.array([Nature.VEGETATIVE])
        self.probability_tables = self.get_probability_tables(self.path)

    @xs.runtime(args=('step'))
    def run_step(self, step):
        if np.any(self.appeared):
            self.has_veg_children_between[self.appeared == 1.] = 0.
            self.nature[self.appeared == 1.] = Nature.VEGETATIVE
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.appeared == 1.) & (self.has_veg_children_within == 0.)):
                    if self.fruiting[gu] == 1.:
                        self.nature[gu] = Nature.FRUITING
                    elif self.flowering[gu] == 1.:
                        self.nature[gu] = Nature.PURE_FLOWER
                    else:
                        self.nature[gu] = Nature.VEGETATIVE
                    index = self.get_factor_values(tbl, gu)
                    probability = tbl[tbl.index == index].probability.values
                    if len(probability):
                        self.has_veg_children_between[gu] = self.rng.binomial(1, probability[0])
