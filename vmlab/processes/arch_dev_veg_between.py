import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, arch_dev_veg_within, arch_dev_rep
from ._base.probability_table import ProbabilityTableProcess


@xs.process
class ArchDevVegBetween(ProbabilityTableProcess):

    rng = xs.global_ref('rng')
    appeared = xs.global_ref('gu_appeared')

    GU = xs.foreign(topology.Topology, 'GU')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')

    has_veg_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'has_veg_children_within')

    flowering = xs.foreign(arch_dev_rep.ArchDevRep, 'flowering')
    flowering_week = xs.foreign(arch_dev_rep.ArchDevRep, 'flowering_week')
    flowering_date = xs.foreign(arch_dev_rep.ArchDevRep, 'flowering_date')
    nb_inflorescences = xs.foreign(arch_dev_rep.ArchDevRep, 'nb_inflorescences')
    fruiting = xs.foreign(arch_dev_rep.ArchDevRep, 'fruiting')
    nb_fruits = xs.foreign(arch_dev_rep.ArchDevRep, 'nb_fruits')
    nature = xs.foreign(arch_dev_rep.ArchDevRep, 'nature')

    tbls_has_veg_children_between = xs.any_object()
    tbls_has_apical_child_between = xs.any_object()
    tbls_burst_date_children_between = xs.any_object()
    tbls_has_lateral_children_between = xs.any_object()
    tbls_nb_lateral_children_between = xs.any_object()

    has_veg_children_between = xs.variable(dims='GU', intent='out')
    has_apical_child_between = xs.variable(dims='GU', intent='out')
    burst_month_children_between = xs.variable(dims='GU', intent='out')
    burst_date_children_between = xs.variable(dims='GU', intent='out')
    has_lateral_children_between = xs.variable(dims='GU', intent='out')
    nb_lateral_children_between = xs.variable(dims='GU', intent='out')

    def initialize(self):

        self.has_veg_children_between = np.zeros(self.GU.shape)
        self.has_apical_child_between = np.zeros(self.GU.shape)
        self.burst_date_children_between = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.burst_month_children_between = np.where(
            self.burst_date_children_between != np.datetime64('NAT'),
            self.burst_date_children_between.astype('datetime64[M]').astype(np.int) % 12 + 1,
            -1
        )
        self.has_lateral_children_between = np.zeros(self.GU.shape)
        self.nb_lateral_children_between = np.zeros(self.GU.shape)

        probability_tables = self.get_probability_tables()

        self.tbls_has_veg_children_between = probability_tables['has_veg_children_between']
        self.tbls_has_apical_child_between = probability_tables['has_apical_child_between']
        self.tbls_burst_date_children_between = probability_tables['burst_date_children_between']
        self.tbls_has_lateral_children_between = probability_tables['has_lateral_children_between']
        self.tbls_nb_lateral_children_between = probability_tables['nb_lateral_children_between']

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        maybe_has_veg_children_between = (self.appeared == 1.) & (self.has_veg_children_within == 0.)
        step_year = step_start.astype('datetime64[D]').item().year

        if np.any(maybe_has_veg_children_between):

            if self.current_cycle in self.tbls_has_veg_children_between:
                gu_indices = np.flatnonzero(maybe_has_veg_children_between)
                tbl = self.tbls_has_veg_children_between[self.current_cycle]
                self.has_veg_children_between[gu_indices] = self.get_binomial(tbl, gu_indices)

            has_veg_children_between = (self.has_veg_children_between == 1.) & maybe_has_veg_children_between

            if np.any(has_veg_children_between):

                gu_indices = np.flatnonzero(has_veg_children_between)

                if self.current_cycle in self.tbls_has_apical_child_between:
                    tbl = self.tbls_has_veg_children_between[self.current_cycle]
                    self.has_apical_child_between[gu_indices] = self.get_binomial(tbl, gu_indices)

                if self.current_cycle in self.probability_tables:
                    self.has_lateral_children_between[  # True if no apical child
                        has_veg_children_between & (self.has_apical_child_between == 0.)
                    ] = 1.
                    tbl = self.probability_tables[self.current_cycle]
                    has_apical_child_between_indices = np.flatnonzero(has_veg_children_between & (self.has_apical_child_between == 1.))
                    self.has_lateral_children_between[has_apical_child_between_indices] = self.get_binomial(tbl, has_apical_child_between_indices)

                if self.current_cycle in self.tbls_nb_lateral_children_between:
                    tbl = self.tbls_nb_lateral_children_between[self.current_cycle]
                    has_lateral_children_between_indices = np.flatnonzero(has_veg_children_between & (self.has_lateral_children_between == 1.))
                    self.nb_lateral_children_between[has_lateral_children_between_indices] = self.get_poisson(tbl, has_lateral_children_between_indices)

                if self.current_cycle in self.tbls_burst_date_children_between:
                    tbl = self.tbls_burst_date_children_between[self.current_cycle]
                    for gu in gu_indices:
                        realization = self.get_multinomial(tbl, gu)
                        if np.any(realization):
                            month = (tbl.columns.to_numpy()[np.nonzero(realization)]).astype(np.int)[0]
                            appearance_month = self.appearance_month[gu]
                            year = step_year if month > appearance_month else step_year + 1
                            self.burst_month_children_between[gu] = month
                            self.burst_date_children_between[gu] = np.datetime64(
                                datetime(year, month, 1)
                            )
