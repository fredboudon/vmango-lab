import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, arch_dev_veg_within, arch_dev_mix
from vmlab.enums import Nature
from ._base.probability_table import ProbabilityTableProcess


@xs.process
class ArchDevRep(ProbabilityTableProcess):

    rng = xs.global_ref('rng')
    gu_appeared = xs.global_ref('gu_appeared')

    GU = xs.foreign(topology.Topology, 'GU')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    doy_begin_flowering = xs.foreign(topology.Topology, 'doy_begin_flowering')

    has_veg_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'has_veg_children_within')
    has_mixed_inflo_children_between = xs.foreign(arch_dev_mix.ArchDevMix, 'has_mixed_inflo_children_between')

    tbls_flowering = xs.any_object()
    tbls_flowering_week = xs.any_object()
    tbls_nb_inflorescences = xs.any_object()
    tbls_fruiting = xs.any_object()
    tbls_nb_fruits = xs.any_object()

    flowering = xs.variable(dims='GU', intent='out')
    flowering_week = xs.variable(dims='GU', intent='out')
    flowering_date = xs.variable(dims='GU', intent='out')
    nb_inflorescences = xs.variable(dims='GU', intent='out')
    fruiting = xs.variable(dims='GU', intent='out')
    nb_fruits = xs.variable(dims='GU', intent='out')
    nature = xs.variable(dims='GU', intent='out')

    def initialize(self):

        self.flowering = np.zeros(self.GU.shape)
        self.flowering_week = np.zeros(self.GU.shape)
        self.flowering_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.nb_inflorescences = np.zeros(self.GU.shape)
        self.fruiting = np.zeros(self.GU.shape)
        self.nb_fruits = np.zeros(self.GU.shape)

        probability_tables = self.get_probability_tables()

        self.tbls_flowering = probability_tables['flowering']
        self.tbls_flowering_week = probability_tables['flowering_week']
        self.tbls_nb_inflorescences = probability_tables['nb_inflorescences']
        self.tbls_fruiting = probability_tables['fruiting']
        self.tbls_nb_fruits = probability_tables['nb_fruits']

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        maybe_flowering = (self.appeared == 1.) & (self.has_veg_children_within == 0.) & (self.has_mixed_inflo_children_between == 0.)

        if np.any(maybe_flowering):

            if self.current_cycle in self.tbls_flowering:
                tbl = self.tbls_flowering[self.current_cycle]
                self.flowering[np.flatnonzero(maybe_flowering)] = self.get_binomial(tbl, np.flatnonzero(maybe_flowering))

            is_flowering = maybe_flowering & (self.flowering == 1.)

            if np.any(is_flowering):

                flowering_indices = np.flatnonzero(is_flowering)
                step_date = step_start.astype('datetime64[D]').item()
                doy = step_date.timetuple().tm_yday

                if self.current_cycle in self.nb_inflorescences:
                    tbl = self.tbls_flowering[self.current_cycle]
                    self.nb_inflorescences[flowering_indices] = self.get_poisson(tbl, flowering_indices)

                if self.current_cycle in self.flowering_week:
                    tbl = self.flowering_week[self.current_cycle]
                    begin_flowering = np.datetime64(
                        datetime(step_date.year if doy < self.doy_begin_flowering else step_date.year + 1, 1, 1)
                    ).astype('datetime64[D]') + self.doy_begin_flowering
                    for gu in flowering_indices:
                        realization = self.get_multinomial(tbl, gu)
                        week = tbl.columns.to_numpy()[np.nonzero(realization)].astype(np.float)[0]
                        self.flowering_week[gu] = week
                        if begin_flowering + week * np.timedelta64(7, 'D') <= step_start:
                            self.flowering_date[gu] = step_start + np.timedelta64(7, 'D')
                        else:
                            self.flowering_date[gu] = begin_flowering + week * np.timedelta64(7, 'D')

                if self.current_cycle in self.tbls_fruiting:
                    tbl = self.tbls_fruiting[self.current_cycle]
                    self.fruiting[flowering_indices] = self.get_binomial(tbl, flowering_indices)

                is_fruiting = is_flowering & (self.fruiting == 1.)

                if np.any(is_fruiting) and self.current_cycle in self.nb_fruits:
                    tbl = self.nb_fruits[self.current_cycle]
                    fruiting_indices = np.nonzero(is_fruiting)
                    self.nb_inflorescences[fruiting_indices] = self.get_poisson(tbl, fruiting_indices)

                self.nature[is_fruiting] = Nature.FRUITING
                self.nature[~is_fruiting & is_flowering] = Nature.PURE_FLOWER
