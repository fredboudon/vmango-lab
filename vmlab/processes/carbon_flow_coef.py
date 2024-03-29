import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph

from ._base.parameter import ParameterizedProcess
from . import (
    topology,
    growth,
    phenology,
    harvest
)


@xs.process
class CarbonFlowCoef(ParameterizedProcess):

    GU = xs.foreign(topology.Topology, 'GU')
    appeared = xs.foreign(topology.Topology, 'appeared')
    adjacency = xs.foreign(topology.Topology, 'adjacency')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')
    gu_stage = xs.foreign(phenology.Phenology, 'gu_stage')
    fruited = xs.foreign(phenology.Phenology, 'fruited')
    harvested = xs.foreign(harvest.Harvest, 'harvested')

    # Must be object because it does only store those GU indices on axis 0 that actually bear a fruit
    # For other purposes there is a full (GUxGU) distance matrix stored in variable 'distances'
    distance_to_fruit = xs.any_object()
    # the only variable that uses nan explicitly (so np.nansum etc. can be used)
    is_in_distance_to_fruit = xs.any_object()
    allocation_share = xs.any_object()
    is_photo_active = xs.variable(dims='GU', intent='out')
    max_distance_to_fruit = xs.variable(
        intent='inout',
        static=True,
        description='Maximum distance (hops) between source and sink GUs',
        default=3
    )
    distances = xs.variable(
        dims=('GU', 'GU_'),
        intent='out',
        description='Distance of each leafy GU to all fruit bearing GUs where no hops <= max_distance_to_fruit',
        attrs={
            'fill_value': np.inf,
            'unit': 'no hops'
        }
    )

    def initialize(self):

        super(CarbonFlowCoef, self).initialize()

        self.distance_to_fruit = np.array([], dtype=np.float32)
        self.is_in_distance_to_fruit = np.array([], dtype=np.bool)
        self.allocation_share = np.array([], dtype=np.float32)
        self.is_photo_active = np.zeros(self.GU.shape, dtype=np.float32)
        self.distances = np.full((self.GU.shape[0], self.GU.shape[0]), np.inf, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):

        is_fruting = (self.nb_fruit > 0.)
        is_leafy = (self.nb_leaf > 0.) & (self.gu_stage >= 4.)

        if np.any(is_fruting):

            if np.any((self.appeared == 1.) | (self.fruited == 1.) | (self.harvested == 1.)):
                self.distance_to_fruit = csgraph.shortest_path(
                    self.adjacency,
                    indices=np.flatnonzero(is_fruting),
                    directed=False
                ).astype(np.float32)
                self.distances[:] = np.inf

            self.distance_to_fruit[self.distance_to_fruit > self.max_distance_to_fruit] = np.inf
            self.distances[np.flatnonzero(is_fruting), :] = self.distance_to_fruit
            self.distance_to_fruit[:, np.flatnonzero(~is_leafy)] = np.inf

            self.is_in_distance_to_fruit = np.isfinite(self.distance_to_fruit).astype(np.float32)
            self.is_in_distance_to_fruit[self.is_in_distance_to_fruit == 0.] = np.nan
            sum_is_in_distance_to_fruit = np.nansum(self.is_in_distance_to_fruit, axis=0)
            self.allocation_share = np.zeros(self.is_in_distance_to_fruit.shape, dtype=np.float32)
            self.allocation_share[:, sum_is_in_distance_to_fruit > 0] = self.is_in_distance_to_fruit[:, sum_is_in_distance_to_fruit > 0] / sum_is_in_distance_to_fruit[sum_is_in_distance_to_fruit > 0]
            self.allocation_share[np.isnan(self.allocation_share)] = 0.
            self.is_photo_active = ((np.nansum(self.is_in_distance_to_fruit, axis=0) > 0) & is_leafy).astype(np.float32)
        else:
            self.distance_to_fruit = np.array([], dtype=np.float32)
            self.is_in_distance_to_fruit = np.array([], dtype=np.bool)
            self.allocation_share = np.array([], dtype=np.float32)
            self.is_photo_active = np.zeros(self.GU.shape, dtype=np.float32)
