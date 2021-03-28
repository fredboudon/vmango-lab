import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph

from ._base.parameter import ParameterizedProcess
from . import (
    topology,
    growth,
    phenology
)


@xs.process
class CarbonAllocation(ParameterizedProcess):

    adjacency = xs.foreign(topology.Topology, 'adjacency')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')
    gu_stage = xs.foreign(phenology.Phenology, 'gu_stage')

    distance_to_fruit = xs.any_object()
    is_in_distance_to_fruit = xs.any_object()
    allocation_share = xs.any_object()
    is_photo_active = xs.variable(dims='GU', intent='out')

    @xs.runtime(args=())
    def run_step(self):

        max_distance_to_fruit = self.parameters.max_distance_to_fruit

        is_fruting = (self.nb_fruit > 0.)
        is_photo_active = (self.nb_leaf > 0.) & (self.gu_stage >= 4.)

        self.distance_to_fruit = csgraph.shortest_path(
            self.adjacency,
            indices=np.flatnonzero(is_fruting),
            directed=False
        ).astype(np.float32)

        self.distance_to_fruit[self.distance_to_fruit > max_distance_to_fruit] = np.inf

        self.is_in_distance_to_fruit = np.isfinite(self.distance_to_fruit)

        # TODO: get rid of zero divide error
        self.allocation_share = np.where(
            np.sum(self.is_in_distance_to_fruit, axis=0) > 0,
            self.is_in_distance_to_fruit / np.sum(self.is_in_distance_to_fruit, axis=0),
            0.
        ).astype(np.float32)
        self.is_photo_active = ((np.sum(self.is_in_distance_to_fruit, axis=0) > 0) & is_photo_active).astype(np.float32)
