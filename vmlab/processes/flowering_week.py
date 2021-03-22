import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, flowering
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class FloweringWeek(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    flowering_week = xs.variable(dims='GU', intent='out')
    flowering_date = xs.variable(dims='GU', intent='inout', groups='arch_dev')

    GU = xs.foreign(topology.Topology, 'GU')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')
    doy_begin_flowering = xs.foreign(topology.Topology, 'doy_begin_flowering')

    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    ancestor_is_apical = xs.foreign(topology.Topology, 'ancestor_is_apical')

    flowering = xs.foreign(flowering.Flowering, 'flowering')

    def initialize(self):
        self.flowering_week = np.full(self.GU.shape, np.nan)
        self.flowering_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.flowering_date[(self.appeared == 1.)] = np.datetime64('NAT')
            self.flowering_week[(self.appeared == 1.)] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                if np.any((self.flowering == 1.) & (self.appeared == 1.)):
                    step_date = step_start.astype('datetime64[D]').item()
                    doy = step_date.timetuple().tm_yday
                    begin_flowering = np.datetime64(datetime(step_date.year if doy < self.doy_begin_flowering else step_date.year + 1, 1, 1)).astype('datetime64[D]') + self.doy_begin_flowering
                    gu_indices = np.flatnonzero((self.flowering == 1.) & (self.appeared == 1.))
                    indices = self.get_indices(tbl, gu_indices)
                    probabilities = tbl.loc[indices.tolist()].values
                    for i, gu_index in enumerate(gu_indices):
                        ps = probabilities[i]
                        if ps.sum() > 0:
                            realization = self.rng.multinomial(1, ps)
                            week = tbl.columns.to_numpy()[np.nonzero(realization)].astype(np.float)[0]
                            self.flowering_week[gu_index] = week
                            if begin_flowering + week * np.timedelta64(7, 'D') <= step_start:
                                self.flowering_date[gu_index] = step_start + np.timedelta64(7, 'D')
                            else:
                                self.flowering_date[gu_index] = begin_flowering + week * np.timedelta64(7, 'D')
