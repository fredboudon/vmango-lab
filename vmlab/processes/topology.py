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

    nb_leaf = xs.variable(
        dims='GU',
        intent='inout'
    )

    nb_inflo = xs.variable(
        dims='GU',
        intent='inout'
    )

    nb_fruit = xs.variable(
        dims='GU',
        intent='inout'
    )

    def initialize(self):

        self.GU = np.array([f'GU{x}' for x in range(self.adjacency.shape[0])], dtype=np.dtype('<U10'))
        self.adjacency = np.array(self.adjacency)
        self.nb_leaf = np.array(self.nb_leaf)
        self.nb_inflo = np.array(self.nb_inflo)

    @xs.runtime(args=())
    def run_step(self):
        pass
