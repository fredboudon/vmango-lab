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
    burst_month_within = xs.variable(dims='GU', intent='out')

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

    def initialize(self):
        self.burst_date_children_within = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.probability_tables = self.get_probability_tables()
        self.burst_month_within = np.where(
            self.burst_date_children_within != np.datetime64('NAT'),
            self.burst_date_children_within.astype('datetime64[M]').astype(np.int) % 12 + 1,
            -1
        )

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        if np.any(self.appeared):
            self.burst_date_children_within[self.appeared == 1.] = np.datetime64('NAT')
            year = step_start.astype('datetime64[D]').item().year
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.has_veg_children_within == 1.) & (self.appeared == 1.)):
                    index = self.get_indices(tbl, np.array(gu))
                    probabilities = tbl.loc[index.tolist()].values.flatten()
                    if probabilities.sum() > 0.:
                        realization = self.rng.multinomial(1, probabilities)
                        month = (tbl.columns.to_numpy()[np.nonzero(realization)]).astype(np.int)[0]
                        appearance_month = self.appearance_month[gu]
                        year = year if month > appearance_month else year + 1
                        self.burst_date_children_within[gu] = np.datetime64(
                            datetime(year, month, 1)
                        )
            self.burst_month_within = np.where(
                self.burst_date_children_within != np.datetime64('NAT'),
                self.burst_date_children_within.astype('datetime64[M]').astype(np.int) % 12 + 1,
                -1
            )
