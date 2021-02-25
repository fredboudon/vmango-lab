import xsimlab as xs
import numpy as np

from . import (
    topology,
    burst_date_children_within, has_veg_children_within,
    has_apical_child_within, has_lateral_children_within, nb_lateral_children_within
)
from ._base import ProbabilityTableBase


@xs.process
class Other(ProbabilityTableBase):

    # path = xs.variable()
    probability_tables = xs.any_object()

    burst_date = xs.variable(dims='GU', intent='inout', groups='archdev')
    has_apical_child = xs.variable(dims='GU', intent='inout', groups='archdev')
    nb_lateral_children = xs.variable(dims='GU', intent='inout', groups='archdev')

    burst_date_children_within = xs.foreign(burst_date_children_within.BurstDateChildrenWithin, 'burst_date_children_within')
    has_veg_children_within = xs.foreign(has_veg_children_within.HasVegChildrenWithin, 'has_veg_children_within')
    has_apical_child_within = xs.foreign(has_apical_child_within.HasApicalChildWithin, 'has_apical_child_within')
    has_lateral_children_within = xs.foreign(has_lateral_children_within.HasLateralChildrenWithin, 'has_lateral_children_within')
    nb_lateral_children_within = xs.foreign(nb_lateral_children_within.NbLateralChildrenWithin, 'nb_lateral_children_within')

    # cycle = xs.foreign(topology.Topology, 'cycle')
    # adjacency = xs.foreign(topology.Topology, 'adjacency')
    # position = xs.foreign(topology.Topology, 'position')
    # nb_fruits = xs.foreign(topology.Topology, 'nb_fruits')
    # bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')

    def initialize(self):
        # self.probability_tables = self.get_probability_tables(self.path)
        self.burst_date = np.array(self.burst_date, dtype='datetime64[D]')
        # self.has_apical_child = np.array([1.])
        # self.nb_lateral_children = np.array([1.])

    @xs.runtime(args=('step',  'step_start', 'step_end', 'step_delta'))
    def run_step(self, step, step_start, step_end, step_delta):
        if np.any(self.appeared):
            self.burst_date[self.appeared == 1.] = np.datetime64('NAT')
            self.has_apical_child[self.appeared == 1.] = 0.
            self.nb_lateral_children[self.appeared == 1.] = 0.

            mask = (self.appeared == 1.) & (self.has_veg_children_within == 1.)
            self.burst_date[mask] = self.burst_date_children_within[mask]
            self.has_apical_child[mask & (self.has_apical_child_within == 1.)] = 1.
            self.nb_lateral_children[mask & (self.has_lateral_children_within == 1.)] = self.nb_lateral_children_within[
                mask & (self.has_lateral_children_within == 1.)
            ]
