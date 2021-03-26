import xsimlab as xs
import numpy as np
from datetime import datetime

from . import topology, arch_dev_veg_within, arch_dev_mix
from vmlab.enums import Nature
from ._base.probability_table import ProbabilityTableProcess


@xs.process
class ArchDevRep(ProbabilityTableProcess):

    GU = xs.foreign(topology.Topology, 'GU')
    seed = xs.foreign(topology.Topology, 'seed')
    appeared = xs.foreign(topology.Topology, 'appeared')
    is_apical = xs.foreign(topology.Topology, 'is_apical')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')
    ancestor_is_apical = xs.foreign(topology.Topology, 'ancestor_is_apical')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    doy_begin_flowering = xs.foreign(topology.Topology, 'doy_begin_flowering')
    is_initially_terminal = xs.foreign(topology.Topology, 'is_initially_terminal')
    sim_start_date = xs.foreign(topology.Topology, 'sim_start_date')

    has_veg_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'has_veg_children_within')
    has_mixed_inflo_children_between = xs.foreign(arch_dev_mix.ArchDevMix, 'has_mixed_inflo_children_between')

    tbls_flowering = None
    tbls_flowering_week = None
    tbls_nb_inflorescences = None
    tbls_fruiting = None
    tbls_nb_fruits = None

    flowering = xs.variable(dims='GU', intent='out')
    flowering_week = xs.variable(dims='GU', intent='out')
    flowering_date = xs.variable(dims='GU', intent='out')
    nb_inflorescences = xs.variable(dims='GU', intent='out')
    fruiting = xs.variable(dims='GU', intent='out')
    nb_fruits = xs.variable(dims='GU', intent='out')
    nature = xs.variable(dims='GU', intent='out')

    def initialize(self):

        super(ArchDevRep, self).initialize()

        self.flowering = np.zeros(self.GU.shape, dtype=np.float32)
        self.flowering_week = np.zeros(self.GU.shape, dtype=np.float32)
        self.flowering_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.nb_inflorescences = np.zeros(self.GU.shape, dtype=np.float32)
        self.fruiting = np.zeros(self.GU.shape, dtype=np.float32)
        self.nb_fruits = np.zeros(self.GU.shape, dtype=np.float32)
        self.nature = np.full(self.GU.shape, Nature.VEGETATIVE, dtype=np.float32)

        probability_tables = self.get_probability_tables()

        self.tbls_flowering = probability_tables['flowering']
        self.tbls_flowering_week = probability_tables['flowering_week']
        self.tbls_nb_inflorescences = probability_tables['nb_inflorescences']
        self.tbls_fruiting = probability_tables['fruiting']
        self.tbls_nb_fruits = probability_tables['nb_fruits']

        self.run_step(-1, self.sim_start_date)

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        appeared = (self.appeared == 1.) if step >= 0 else (self.is_initially_terminal == 1.)

        maybe_flowering = appeared & (self.has_veg_children_within == 0.) & (self.has_mixed_inflo_children_between == 0.)

        if np.any(maybe_flowering):

            if self.current_cycle in self.tbls_flowering:
                tbl = self.tbls_flowering[self.current_cycle]
                self.flowering[np.flatnonzero(maybe_flowering)] = self.get_binomial(tbl, np.flatnonzero(maybe_flowering))

            is_flowering = maybe_flowering & (self.flowering == 1.)

            if np.any(is_flowering):

                flowering_indices = np.flatnonzero(is_flowering)
                step_date = step_start.astype('datetime64[D]').item()
                doy = step_date.timetuple().tm_yday

                if self.current_cycle in self.tbls_flowering:
                    tbl = self.tbls_flowering[self.current_cycle]
                    self.nb_inflorescences[flowering_indices] = self.get_poisson(tbl, flowering_indices)

                if self.current_cycle in self.tbls_flowering_week:
                    tbl = self.tbls_flowering_week[self.current_cycle]
                    begin_flowering = np.datetime64(
                        datetime(step_date.year if doy < self.doy_begin_flowering else step_date.year + 1, 1, 1)
                    ).astype('datetime64[D]') + self.doy_begin_flowering
                    for gu in flowering_indices:
                        realization = self.get_multinomial(tbl, gu)
                        if np.any(realization):
                            week = tbl.columns.to_numpy()[np.nonzero(realization)].astype(np.float)[0]
                            self.flowering_week[gu] = week
                            if begin_flowering + week * np.timedelta64(7, 'D') <= step_start:
                                self.flowering_date[gu] = step_start + np.timedelta64(7, 'D')
                            else:
                                self.flowering_date[gu] = begin_flowering + week * np.timedelta64(7, 'D')
                        else:
                            self.flowering_week[gu] = 0
                            self.flowering_date[gu] = np.datetime64('NAT')

                if self.current_cycle in self.tbls_fruiting:
                    tbl = self.tbls_fruiting[self.current_cycle]
                    self.fruiting[flowering_indices] = self.get_binomial(tbl, flowering_indices)

                is_fruiting = is_flowering & (self.fruiting == 1.)

                if np.any(is_fruiting) and self.current_cycle in self.tbls_nb_fruits:
                    tbl = self.tbls_nb_fruits[self.current_cycle]
                    fruiting_indices = np.nonzero(is_fruiting)
                    self.nb_fruits[fruiting_indices] = self.get_poisson(tbl, fruiting_indices)

                self.nature[is_fruiting] = Nature.FRUITING
                self.nature[~is_fruiting & is_flowering] = Nature.PURE_FLOWER
