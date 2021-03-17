import xsimlab as xs
import numpy as np

from . import topology, flowering
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class NbInflorescences(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    nb_inflorescences = xs.variable(dims='GU', intent='out')

    GU = xs.foreign(topology.Topology, 'GU')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')

    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    ancestor_nature = xs.foreign(topology.Topology, 'ancestor_nature')

    flowering = xs.foreign(flowering.Flowering, 'flowering')

    def initialize(self):
        self.nb_inflorescences = np.zeros(self.GU.shape)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):
            self.nb_inflorescences[self.appeared == 1.] = 0.
            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                if np.any((self.appeared == 1.) & (self.flowering == 1.)):
                    gu_indices = np.flatnonzero((self.appeared == 1.) & (self.flowering == 1.))
                    indices = self.get_indices(tbl, gu_indices)
                    lam = tbl.loc[indices.tolist()].values.flatten()
                    self.nb_inflorescences[gu_indices] = np.round(self.rng.poisson(lam, lam.shape) + 1., 0)
