import xsimlab as xs
import numpy as np

from . import (
    topology,
    arch_dev_veg_within,
    arch_dev_veg_between,
    arch_dev_rep
)


@xs.process
class ArchDev:
    """Arch Dev process only depending on input (deterministic)
    """

    GU = xs.foreign(topology.Topology, 'GU')

    pot_burst_date = xs.variable(dims='GU', intent='inout', groups='arch_dev', encoding={'fill_value': 'NaT'})
    pot_flowering_date = xs.variable(dims='GU', intent='inout', groups='arch_dev', encoding={'fill_value': 'NaT'})
    pot_nature = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_has_apical_child = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nb_lateral_children = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nb_inflo = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    pot_nb_fruit = xs.variable(dims='GU', intent='inout', groups='arch_dev')

    def initialize(self):
        self.pot_burst_date = np.array(self.pot_burst_date, dtype='datetime64[ns]')
        self.pot_flowering_date = np.array(self.pot_flowering_date, dtype='datetime64[ns]')
        self.pot_nature = np.array(self.pot_nature, dtype=np.float32)
        self.pot_has_apical_child = np.array(self.pot_has_apical_child, dtype=np.float32)
        self.pot_nb_lateral_children = np.array(self.pot_nb_lateral_children, dtype=np.float32)
        self.pot_nb_inflo = np.array(self.pot_nb_inflo, dtype=np.float32)
        self.pot_nb_fruit = np.array(self.pot_nb_fruit, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):
        pass


@xs.process
class ArchDevStochastic(ArchDev):
    """Arch Dev process that depends on all stochastic arch dev processes
    """

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
        super(ArchDevStochastic, self).initialize()
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
