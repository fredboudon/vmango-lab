import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, has_veg_children_within
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class BurstDateChildrenWithin(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    burst_date_children_within = xs.variable(dims='GU', intent='out')

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
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')

    def initialize(self):
        self.burst_date_children_within = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        if np.any(self.appeared):
            self.burst_date_children_within[self.appeared == 1.] = np.datetime64('NAT')
            gu_mask = (self.has_veg_children_within == 1.) & (self.appeared == 1.)
            if np.any(gu_mask):
                gu_indices = np.nonzero(gu_mask)
                year = step_start.astype('datetime64[D]').item().year
                if self.current_cycle in self.probability_tables:
                    tbl = self.probability_tables[self.current_cycle]
                    indices = self.get_indices(tbl, gu_indices)
                    probabilities = tbl.loc[indices.tolist()].values
                    realization = np.array([self.rng.multinomial(1, ps) for ps in probabilities])
                    month = (tbl.columns.to_numpy()[np.argwhere(realization)[:, 1]]).astype(np.int)
                    self.burst_date_children_within[gu_indices] = np.where(
                        month > self.appearance_month[gu_indices],
                        [np.datetime64(datetime(year, month, 1)) for month in month],
                        [np.datetime64(datetime(year + 1, month, 1)) for month in month],
                    )
