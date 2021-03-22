import xsimlab as xs
import numpy as np

from . import topology, flowering
from ._base.probability_table import BaseProbabilityTableProcess


@xs.process
class FloweringWeek(BaseProbabilityTableProcess):

    rng = xs.global_ref('rng')

    probability_tables = xs.any_object()

    flowering_week = xs.variable(dims='GU', intent='out')

    GU = xs.foreign(topology.Topology, 'GU')
    current_cycle = xs.foreign(topology.Topology, 'current_cycle')
    cycle = xs.foreign(topology.Topology, 'cycle')
    seed = xs.foreign(topology.Topology, 'seed')
    adjacency = xs.foreign(topology.Topology, 'adjacency')
    ancestor = xs.foreign(topology.Topology, 'ancestor')
    position = xs.foreign(topology.Topology, 'position')
    bursted = xs.foreign(topology.Topology, 'bursted')
    appeared = xs.foreign(topology.Topology, 'appeared')
    appearance_month = xs.foreign(topology.Topology, 'appearance_month')
    ancestor_is_apical = xs.foreign(topology.Topology, 'ancestor_is_apical')

    flowering = xs.foreign(flowering.Flowering, 'flowering')

    def initialize(self):
        self.flowering_week = np.full(self.GU.shape, np.nan)
        self.probability_tables = self.get_probability_tables()

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):
        if np.any(self.appeared):

            if self.current_cycle in self.probability_tables:
                tbl = self.probability_tables[self.current_cycle]
                if np.any((self.flowering == 1.) & (self.appeared == 1.)):
                    gu_indices = np.nonzero((self.flowering == 1.) & (self.appeared == 1.))
                    indices = self.get_indices(tbl, gu_indices)
                    probabilities = tbl.loc[indices.tolist()].values
                    realization = np.array([self.rng.multinomial(1, ps) for ps in probabilities])
                    week = (tbl.columns.to_numpy()[np.argwhere(realization)[:, 1]]).astype(np.float)
                    self.flowering_week[gu_indices] = week
