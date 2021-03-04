import xsimlab as xs
import numpy as np


@xs.process
class Topology:

    GU = xs.index(
        dims='GU',
        global_name='GU'
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
        self.adjacency = self.adjacency.todense()
        self.nb_leaves_gu = np.array(self.nb_leaves_gu, dtype=np.int64)

    @xs.runtime(args=())
    def run_step(self):
        pass
