import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph

from . import topology


@xs.process
class BaseCarbonAllocation:

    GU = xs.foreign(topology.Topology, 'GU')

    carbon_allocation = xs.variable(
        dims=('GU', 'GU'),
        intent='out'
    )


@xs.process
class CarbonAllocationIdentity(BaseCarbonAllocation):

    def initialize(self):

        self.carbon_allocation = np.identity(self.GU.shape[0])

    @xs.runtime(args=())
    def run_step(self):
        pass


@xs.process
class CarbonAllocationByDistance(CarbonAllocationIdentity):

    adjacency = xs.foreign(topology.Topology, 'adjacency')

    distances = xs.variable(
        dims=('GU', 'GU'),
        intent='out'
    )

    @xs.runtime(args=())
    def run_step(self):

        self.distances = csgraph.shortest_path(self.adjacency)

        # self.carbon_allocation = ??
