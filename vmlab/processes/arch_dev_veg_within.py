import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology
from ._base.probability_table import ProbabilityTableProcess


@xs.process
class ArchDevVegWithin(ProbabilityTableProcess):

    rng = xs.global_ref('rng')

    GU = xs.foreign(topology.Topology, 'GU')
    appeared = xs.foreign(topology.Topology, 'appeared')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    is_apical = xs.foreign(topology.Topology, 'is_apical')
    ancestor_is_apical = xs.foreign(topology.Topology, 'ancestor_is_apical')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    tbls_has_veg_children_within = None
    tbls_has_apical_child_within = None
    tbls_burst_date_children_within = None
    tbls_has_lateral_children_within = None
    tbls_nb_lateral_children_within = None

    has_veg_children_within = xs.variable(dims='GU', intent='out')
    has_apical_child_within = xs.variable(dims='GU', intent='out')
    burst_month_children_within = xs.variable(dims='GU', intent='out')
    burst_date_children_within = xs.variable(dims='GU', intent='out')
    has_lateral_children_within = xs.variable(dims='GU', intent='out')
    nb_lateral_children_within = xs.variable(dims='GU', intent='out')

    def initialize(self):

        super(ArchDevVegWithin, self).initialize()

        self.has_veg_children_within = np.zeros(self.GU.shape)
        self.has_apical_child_within = np.zeros(self.GU.shape)
        self.burst_date_children_within = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.burst_month_children_within = np.where(
            self.burst_date_children_within != np.datetime64('NAT'),
            self.burst_date_children_within.astype('datetime64[M]').astype(np.int) % 12 + 1,
            -1
        )
        self.has_lateral_children_within = np.zeros(self.GU.shape)
        self.nb_lateral_children_within = np.zeros(self.GU.shape)

        probability_tables = self.get_probability_tables()

        self.tbls_has_veg_children_within = probability_tables['has_veg_children_within']
        self.tbls_has_apical_child_within = probability_tables['has_apical_child_within']
        self.tbls_burst_date_children_within = probability_tables['burst_date_children_within']
        self.tbls_has_lateral_children_within = probability_tables['has_lateral_children_within']
        self.tbls_nb_lateral_children_within = probability_tables['nb_lateral_children_within']

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        appeared = self.appeared == 1.
        step_year = step_start.astype('datetime64[D]').item().year

        if np.any(appeared):

            if self.current_cycle in self.tbls_has_veg_children_within:
                gu_indices = np.flatnonzero(appeared)
                tbl = self.tbls_has_veg_children_within[self.current_cycle]
                self.has_veg_children_within[gu_indices] = self.get_binomial(tbl, gu_indices)

            has_veg_children_within = (self.has_veg_children_within == 1.) & appeared

            if np.any(has_veg_children_within):

                gu_indices = np.flatnonzero(has_veg_children_within)

                if self.current_cycle in self.tbls_has_apical_child_within:
                    tbl = self.tbls_has_veg_children_within[self.current_cycle]
                    self.has_apical_child_within[gu_indices] = self.get_binomial(tbl, gu_indices)

                if self.current_cycle in self.tbls_has_lateral_children_within:
                    self.has_lateral_children_within[  # True if no apical child
                        has_veg_children_within & (self.has_apical_child_within == 0.)
                    ] = 1.
                    tbl = self.tbls_has_lateral_children_within[self.current_cycle]
                    has_apical_child_within_indices = np.flatnonzero(has_veg_children_within & (self.has_apical_child_within == 1.))
                    self.has_lateral_children_within[has_apical_child_within_indices] = self.get_binomial(tbl, has_apical_child_within_indices)

                if self.current_cycle in self.tbls_nb_lateral_children_within:
                    tbl = self.tbls_nb_lateral_children_within[self.current_cycle]
                    has_lateral_children_within_indices = np.flatnonzero(has_veg_children_within & (self.has_lateral_children_within == 1.))
                    self.nb_lateral_children_within[has_lateral_children_within_indices] = self.get_poisson(tbl, has_lateral_children_within_indices)

                if self.current_cycle in self.tbls_burst_date_children_within:
                    tbl = self.tbls_burst_date_children_within[self.current_cycle]
                    for gu in gu_indices:
                        realization = self.get_multinomial(tbl, gu)
                        if np.any(realization):
                            month = (tbl.columns.to_numpy()[np.nonzero(realization)]).astype(np.int)[0]
                            appearance_month = self.appearance_month[gu]
                            year = step_year if month > appearance_month else step_year + 1
                            self.burst_month_children_within[gu] = month
                            self.burst_date_children_within[gu] = np.datetime64(
                                datetime(year, month, 1)
                            )
