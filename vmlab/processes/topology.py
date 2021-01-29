import xsimlab as xs
import numpy as np

from . import parameters
from .base import BaseGrowthUnitProcess


@xs.process
class Topology(BaseGrowthUnitProcess):

    params = xs.foreign(parameters.Parameters, 'topology')

    bursts = xs.group_dict('bursts')

    GU = xs.index(
        dims='GU',
        global_name='GU'
    )

    is_apical = xs.variable(
        dims=('GU'),
        intent='inout',
        static=True
    )

    adjacency = xs.variable(
        dims=('GU', 'GU'),
        intent='inout'
    )

    nb_leaves_gu = xs.variable(
        dims=('GU'),
        intent='inout'
    )

    def initialize(self):

        # self.GU = np.arange(self.adjacency.shape[0], dtype=np.int64)
        self.GU = np.array([f'GU{x}' for x in range(self.adjacency.shape[0])], dtype=np.dtype('<U10'))
        # TODO: Currently no sparse support in simlab
        # https://github.com/benbovy/xarray-simlab/issues/165
        self.adjacency = self.adjacency.todense()
        self.nb_leaves_gu = np.array(self.nb_leaves_gu, dtype=np.int64)

    def step(self, nsteps, step, step_start, step_end, step_delta):

        nb_bursted = np.count_nonzero(self.bursts[('gu_burst', 'gu_bursted')])
        if nb_bursted > 0:
            self.GU = np.append(self.GU, [f'NEW{x}' for x in range(nb_bursted)])
            self._resize(step)
            # update adj matrix for each new GU
            parent_idxs = np.nonzero(self.bursts[('gu_burst', 'gu_bursted')])[0]
            for i in range(nb_bursted):
                parent_idx = parent_idxs[i]
                child_idx = self.GU.shape[0] - nb_bursted + i
                self.adjacency[parent_idx, child_idx] = True
