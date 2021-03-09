import xsimlab as xs
import numpy as np

from . import (
    topology,
    has_veg_children_within,
    burst_date_children_within,
    has_apical_child_within,
    has_lateral_children_within,
    nb_lateral_children_within,
    has_mixed_inflo_children_between,
    has_veg_children_between,
    burst_date_children_between,
    has_apical_child_between,
    has_lateral_children_between,
    nb_lateral_children_between,
    flowering,
    flowering_week,
    nb_inflorescences,
    fruiting,
    nb_fruits
)


@xs.process
class ArchDev:

    # path = xs.variable()
    probability_tables = xs.any_object()

    burst_date = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    has_apical_child = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    nb_lateral_children = xs.variable(dims='GU', intent='inout', groups='arch_dev')
    nature = xs.variable(dims='GU', intent='inout', groups='arch_dev')

    burst_date_children_within = xs.foreign(burst_date_children_within.BurstDateChildrenWithin, 'burst_date_children_within')
    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')
    has_apical_child_within = xs.foreign(has_apical_child_within.HasApicalChildWithin, 'has_apical_child_within')
    has_lateral_children_within = xs.foreign(has_lateral_children_within.HasLateralChildrenWithin, 'has_lateral_children_within')
    nb_lateral_children_within = xs.foreign(nb_lateral_children_within.NbLateralChildrenWithin, 'nb_lateral_children_within')

    has_mixed_inflo_children_between = xs.foreign(has_mixed_inflo_children_between.HasMixedInfloChildrenBetween, 'has_mixed_inflo_children_between')

    burst_date_children_between = xs.foreign(burst_date_children_between.BurstDateChildrenBetween, 'burst_date_children_between')
    nature_ = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'nature')
    has_veg_children_between = xs.foreign(has_veg_children_between.HasVegChildrenBetween, 'has_veg_children_between')
    has_apical_child_between = xs.foreign(has_apical_child_between.HasApicalChildBetween, 'has_apical_child_between')
    has_lateral_children_between = xs.foreign(has_lateral_children_between.HasLateralChildrenBetween, 'has_lateral_children_between')
    nb_lateral_children_between = xs.foreign(nb_lateral_children_between.NbLateralChildrenBetween, 'nb_lateral_children_between')

    flowering = xs.foreign(flowering.Flowering, 'flowering')
    flowering_week = xs.foreign(flowering_week.FloweringWeek, 'flowering_week')
    nb_inflorescences = xs.foreign(nb_inflorescences.NbInflorescences, 'nb_inflorescences')
    fruiting = xs.foreign(fruiting.Fruiting, 'fruiting')
    nb_fruits = xs.foreign(nb_fruits.NbFruits, 'nb_fruits')

    # cycle = xs.foreign(topology.Topology, 'cycle')
    # adjacency = xs.foreign(topology.Topology, 'adjacency')
    # position = xs.foreign(topology.Topology, 'position')
    # nb_fruits = xs.foreign(topology.Topology, 'nb_fruits')
    # bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')

    def initialize(self):
        self.burst_date = np.array(self.burst_date, dtype='datetime64[D]')
        self.has_apical_child = np.array(self.has_apical_child)
        self.nb_lateral_children = np.array(self.nb_lateral_children)
        self.nature = np.array(self.nature)

    @xs.runtime(args=('step',  'step_start', 'step_end', 'step_delta'))
    def run_step(self, step, step_start, step_end, step_delta):
        if np.any(self.appeared):
            self.burst_date[self.appeared == 1.] = np.datetime64('NAT')
            self.has_apical_child[self.appeared == 1.] = 0.
            self.nb_lateral_children[self.appeared == 1.] = 0.

            mask_within = (self.appeared == 1.) & (self.has_veg_children_within == 1.)
            self.burst_date[mask_within] = self.burst_date_children_within[mask_within]
            self.has_apical_child[mask_within & (self.has_apical_child_within == 1.)] = 1.
            self.nb_lateral_children[mask_within & (self.has_lateral_children_within == 1.)] = self.nb_lateral_children_within[
                mask_within & (self.has_lateral_children_within == 1.)
            ]

            mask_between = (self.appeared == 1.) & (self.has_veg_children_between == 1.)
            self.burst_date[mask_between] = self.burst_date_children_between[mask_between]
            self.has_apical_child[mask_between & (self.has_apical_child_between == 1.)] = 1.
            self.nb_lateral_children[mask_between & (self.has_lateral_children_between == 1.)] = self.nb_lateral_children_between[
                mask_between & (self.has_lateral_children_between == 1.)
            ]

            self.nature[self.appeared == 1.] = self.nature_[self.appeared == 1.]
