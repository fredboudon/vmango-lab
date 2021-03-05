import xsimlab as xs

from vmlab.processes.architectural_development import (
    topology,
    has_veg_children_within,
    has_apical_child_within,
    burst_date_children_within,
    has_lateral_children_within,
    nb_lateral_children_within,
    has_mixed_inflo_children_between,
    has_apical_child_between,
    burst_date_children_between,
    has_lateral_children_between,
    nb_lateral_children_between,
    flowering,
    flowering_week,
    nb_inflorescences,
    fruiting,
    nb_fruits,
    has_veg_children_between,
    arch_dev
)


arch_dev = xs.Model({
    'topology': topology.Topology,
    'has_veg_children_within': has_veg_children_within.HasVegChildrenWithin,
    'has_apical_child_within': has_apical_child_within.HasApicalChildWithin,
    'burst_date_children_within': burst_date_children_within.BurstDateChildrenWithin,
    'has_lateral_children_within': has_lateral_children_within.HasLateralChildrenWithin,
    'nb_lateral_children_within': nb_lateral_children_within.NbLateralChildrenWithin,
    'has_mixed_inflo_children_between': has_mixed_inflo_children_between.HasMixedInfloChildrenBetween,
    'has_apical_child_between': has_apical_child_between.HasApicalChildBetween,
    'burst_date_children_between': burst_date_children_between.BurstDateChildrenBetween,
    'has_lateral_children_between': has_lateral_children_between.HasLateralChildrenBetween,
    'nb_lateral_children_between': nb_lateral_children_between.NbLateralChildrenBetween,
    'flowering': flowering.Flowering,
    'flowering_week': flowering_week.FloweringWeek,
    'nb_inflorescences': nb_inflorescences.NbInflorescences,
    'fruiting': fruiting.Fruiting,
    'nb_fruits': nb_fruits.NbFruits,
    'has_veg_children_between': has_veg_children_between.HasVegChildrenBetween,
    'arch_dev': arch_dev.ArchDev
})
