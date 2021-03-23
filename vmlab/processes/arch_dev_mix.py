import xsimlab as xs
import numpy as np

from . import topology, arch_dev_veg_within
from ._base.probability_table import ProbabilityTableProcess


@xs.process
class ArchDevMix(ProbabilityTableProcess):

    rng = xs.global_ref('rng')
    gu_appeared = xs.global_ref('gu_appeared')

    GU = xs.foreign(topology.Topology, 'GU')
    has_veg_children_within = xs.foreign(arch_dev_veg_within.ArchDevVegWithin, 'has_veg_children_within')

    tbls_has_mixed_inflo_children_between = xs.any_object()
    has_mixed_inflo_children_between = xs.variable(dims='GU', intent='out')

    def initialize(self):

        self.has_mixed_inflo_children_between = np.zeros(self.GU.shape)

        probability_tables = self.get_probability_tables()

        self.tbls_has_mixed_inflo_children_between = probability_tables['has_mixed_inflo_children_between']

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        appeared = self.appeared == 1.

        if np.any(appeared & ~self.has_veg_children_within):

            gu_indices = np.flatnonzero(appeared & ~self.has_veg_children_within)

            if self.current_cycle in self.tbls_has_mixed_inflo_children_between:
                tbl = self.tbls_has_mixed_inflo_children_between[self.current_cycle]
                self.has_mixed_inflo_children_between[gu_indices] = self.binomial(tbl, gu_indices)
