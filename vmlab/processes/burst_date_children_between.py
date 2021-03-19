import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, has_veg_children_between
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class BurstDateChildrenBetween(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    burst_date_children_between = xs.variable(dims='GU', intent='out')

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
    month_begin_veg_cycle = xs.foreign(topology.Topology, 'month_begin_veg_cycle')

    nature = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'nature')
    has_veg_children_between = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'has_veg_children_between')

    def initialize(self):
        self.burst_date_children_between = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            year = step_start.astype('datetime64[D]').item().year
            self.burst_date_children_between[self.appeared == 1.] = np.datetime64('NAT')
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                for gu in np.flatnonzero((self.has_veg_children_between == 1.) & (self.appeared == 1.)):
                    index = self.get_indices(tbl, np.array(gu))
                    probabilities = tbl.loc[index.tolist()].values.flatten()
                    if probabilities.sum() > 0.:
                        realization = self.rng.multinomial(1, probabilities)
                        # in case of several results ('101-102-..') we choose the first
                        realization = int(tbl.columns.to_numpy()[np.nonzero(realization)][0].split('-')[0])
                        month = realization % 100
                        if month < self.month_begin_veg_cycle:
                            year = year + 1
                        if realization // 100 > 1:
                            year = year + (realization // 100) - 1
                        self.burst_date_children_between[gu] = np.datetime64(datetime(
                            year,
                            month,
                            1
                        ))
