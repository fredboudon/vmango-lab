import xsimlab as xs
import numpy as np

from . import parameters


@xs.process
class Topology():

    params = xs.foreign(parameters.Parameters, 'topology')

    GU = xs.index(
        dims=('GU')
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

    @xs.runtime(args=('step'))
    def run_step(self, step):
        pass
