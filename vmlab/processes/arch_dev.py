import xsimlab as xs
import numpy as np

from vmlab.enums import Nature

from . import (
    topology,
    arch_dev_veg_within,
    arch_dev_veg_between,
    arch_dev_rep
)


@xs.process
class ArchDev:

    pot_burst_date = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_flowering_date = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nature = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_has_apical_child = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nb_lateral_children = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nb_inflo = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nb_fruit = xs.variable(dims='GU', intent='inout', groups='arch_dev')

    GU = xs.foreign(topology.Topology, 'GU')
    appeared = xs.foreign(topology.Topology, 'appeared')
    is_initially_terminal = xs.foreign(topology.Topology, 'is_initially_terminal')
    sim_start_date = xs.foreign(topology.Topology, 'sim_start_date')

    has_veg_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'has_veg_children_within')
    burst_date_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'burst_date_children_within')
    has_apical_child_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'has_apical_child_within')
    nb_lateral_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'nb_lateral_children_within')

    flowering_date = xs.foreign(arch_dev_rep.ArchDevRep, 'flowering_date')
    nb_inflorescences = xs.foreign(arch_dev_rep.ArchDevRep, 'nb_inflorescences')
    nb_fruits = xs.foreign(arch_dev_rep.ArchDevRep, 'nb_fruits')
    nature = xs.foreign(arch_dev_rep.ArchDevRep, 'nature')

    has_veg_children_between = xs.foreign(arch_dev_veg_between.ArchDevVegBetween, 'has_veg_children_between')
    burst_date_children_between = xs.foreign(arch_dev_veg_between.ArchDevVegBetween, 'burst_date_children_between')
    has_apical_child_between = xs.foreign(arch_dev_veg_between.ArchDevVegBetween, 'has_apical_child_between')
    nb_lateral_children_between = xs.foreign(arch_dev_veg_between.ArchDevVegBetween, 'nb_lateral_children_between')

    def initialize(self):

        self.pot_burst_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.pot_flowering_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.pot_nature = np.full(self.GU.shape, Nature.VEGETATIVE)
        self.pot_has_apical_child = np.full(self.GU.shape, 0.)
        self.pot_nb_lateral_children = np.full(self.GU.shape, 0.)
        self.pot_nb_inflo = np.full(self.GU.shape, 0.)
        self.pot_nb_fruit = np.full(self.GU.shape, 0.)

        self.run_step(-1)

    @xs.runtime(args=('step'))
    def run_step(self, step):

        appeared = (self.appeared == 1.) if step >= 0 else (self.is_initially_terminal == 1.)

        if np.any(appeared):

            within = np.flatnonzero(appeared & (self.has_veg_children_within == 1.))
            self.pot_burst_date[within] = self.burst_date_children_within[within]
            self.pot_has_apical_child[within] = self.has_apical_child_within[within]
            self.pot_nb_lateral_children[within] = self.nb_lateral_children_within[within]

            between = np.flatnonzero(appeared & (self.has_veg_children_between == 1.))
            self.pot_burst_date[between] = self.burst_date_children_between[between]
            self.pot_has_apical_child[between] = self.has_apical_child_between[between]
            self.pot_nb_lateral_children[between] = self.nb_lateral_children_between[between]

            appeared = np.flatnonzero(appeared)
            self.pot_nature[appeared] = self.nature[appeared]
            self.pot_flowering_date[appeared] = self.flowering_date[appeared]
            self.pot_nb_fruit[appeared] = self.nb_fruits[appeared]
            self.pot_nb_inflo[appeared] = self.nb_inflorescences[appeared]
