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

    nb_leaves = xs.variable(
        dims=('GU'),
        intent='inout'
    )

    def initialize(self):

        self.GU = np.array([f'Branch{x}' for x in range(self.adjacency.shape[0])], dtype=np.dtype('<U10'))
        self.adjacency = np.array(self.adjacency)
        self.nb_leaves = np.array(self.nb_leaves)

    @xs.runtime(args=())
    def run_step(self):
        pass
