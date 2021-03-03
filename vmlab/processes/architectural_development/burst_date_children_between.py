import xsimlab as xs
import numpy as np

from . import topology, has_veg_children_between
from ._base import BaseProbabilityTable


@xs.process
class BurstDateChildrenBetween(BaseProbabilityTable):

    rng = xs.global_ref('rng')

    path = xs.variable()
    probability_tables = xs.any_object()

    burst_date_children_between = xs.variable(dims='GU', intent='out')

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

    nature = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'nature')
    has_veg_children_between = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'has_veg_children_between')

    def initialize(self):
        self.burst_date_children_between = np.array([], dtype='U')
        self.probability_tables = self.get_probability_tables(self.path)

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):

            self.burst_date_children_between[self.appeared == 1.] = ''
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.has_veg_children_between == 1.) & (self.appeared == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probabilities = tbl[tbl.index == index].values.flatten()
                    if len(probabilities):
                        realization = np.flatnonzero(self.rng.multinomial(1, probabilities))[0]
                        burst_date_children_between = int(tbl.columns[realization])
                        self.burst_date_children_between[gu] = burst_date_children_between
