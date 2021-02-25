import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, has_veg_children_within
from ._base import ProbabilityTableBase


@xs.process
class BurstDateChildrenWithin(ProbabilityTableBase):

    rng = xs.global_ref('rng')

    path = xs.variable()
    probability_tables = xs.any_object()

    burst_date_children_within = xs.variable(dims='GU', intent='out')

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
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')

    def initialize(self):
        self.burst_date_children_within = np.array([], dtype='datetime64[D]')
        self.probability_tables = self.get_probability_tables(self.path)

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            year = step_start.astype('datetime64[D]').item().year
            self.burst_date_children_within[self.appeared == 1.] = np.datetime64('NAT')
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.has_veg_children_within == 1.) & (self.appeared == 1.)):
                    index = self.get_factor_values(tbl, gu)
                    probabilities = tbl[tbl.index == index].values.flatten()
                    if len(probabilities):
                        realization = np.flatnonzero(self.rng.multinomial(1, probabilities))[0]
                        month = int(tbl.columns[realization])
                        self.burst_date_children_within[gu] = np.datetime64(datetime(
                            year if month > self.appearance_month[gu] else year + 1,
                            month,
                            1
                        ))
