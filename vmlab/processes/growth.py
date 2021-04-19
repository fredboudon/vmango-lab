import xsimlab as xs
import numpy as np
import zarr
import typing

from . import topology, phenology, appearance
from ._base.parameter import ParameterizedProcess


@xs.process
class Growth(ParameterizedProcess):
    """Compute the current length, radius of entities
    """

    rng = None

    GU = xs.foreign(topology.Topology, 'GU')
    nb_descendants = xs.foreign(topology.Topology, 'nb_descendants')
    seed = xs.foreign(topology.Topology, 'seed')

    gu_growth_tts = xs.foreign(phenology.Phenology, 'gu_growth_tts')
    leaf_growth_tts = xs.foreign(phenology.Phenology, 'leaf_growth_tts')
    inflo_growth_tts = xs.foreign(phenology.Phenology, 'inflo_growth_tts')
    gu_stage = xs.foreign(phenology.Phenology, 'gu_stage')
    inflo_stage = xs.foreign(phenology.Phenology, 'inflo_stage')
    nb_gu_stage = xs.foreign(phenology.Phenology, 'nb_gu_stage')
    nb_inflo_stage = xs.foreign(phenology.Phenology, 'nb_inflo_stage')

    final_length_gu = xs.foreign(appearance.Appearance, 'final_length_gu')
    nb_internode = xs.foreign(appearance.Appearance, 'nb_internode')
    final_length_leaves = xs.foreign(appearance.Appearance, 'final_length_leaves')
    final_length_inflos = xs.foreign(appearance.Appearance, 'final_length_inflos')
    appeared = xs.foreign(appearance.Appearance, 'appeared')
    any_is_growing = xs.variable(intent='out', groups='growth')

    radius_gu = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )

    length_gu = xs.variable(
        dims=('GU'),
        intent='out',
        groups='growth'
    )

    length_leaves = xs.variable(
        dims=('GU'),
        intent='out',
        groups='growth',
        encoding={
            'object_codec': zarr.JSON()
        }
    )

    length_inflos = xs.variable(
        dims=('GU'),
        intent='out',
        groups='growth',
        encoding={
            'object_codec': zarr.JSON()
        }
    )

    radius_inflo = xs.variable(
        dims=('GU'),
        intent='out',
        groups='growth'
    )

    nb_leaf = xs.variable(
        dims='GU',
        intent='out',
        groups='growth'
    )

    def get_length_inflos(self, final_length_inflos: typing.List[float], inflo_growth_tts, params):
        final_length_inflos = np.array(final_length_inflos)
        return (final_length_inflos / (1. + np.exp(-(inflo_growth_tts - params.t_ip_inflo) / params.B_inflo))).tolist()

    def get_length_leaves(self, final_length_leaves: typing.List[float], leaf_growth_tts, params):
        final_length_leaves = np.array(final_length_leaves)
        max_growth_rate = -0.0188725 + 0.0147985 * final_length_leaves * 4
        B = final_length_leaves / max_growth_rate
        return (final_length_leaves / (1. + np.exp(-(leaf_growth_tts - params.t_ip_leaf) / B))).tolist()

    def initialize(self):

        super(Growth, self).initialize()

        self.rng = np.random.default_rng(seed=self.seed)

        params = self.parameters
        radius_exponent_gu = params.radius_exponent_gu
        radius_coefficient_gu = params.radius_coefficient_gu
        max_leafy_diameter_gu = params.max_leafy_diameter_gu
        params.t_ip_gu = self.rng.normal(params.t_ip_gu_mean, params.t_ip_gu_sd)

        # np.array of type list
        self.get_length_inflos = np.vectorize(self.get_length_inflos, otypes=[object], excluded={'params'})
        self.get_length_leaves = np.vectorize(self.get_length_leaves, otypes=[object], excluded={'leaf_growth_tts', 'params'})

        self.radius_gu = (radius_coefficient_gu * (self.nb_descendants + 1) ** radius_exponent_gu).astype(np.float32)
        self.radius_inflo = np.zeros(self.GU.shape, dtype=np.float32)
        self.length_gu = self.final_length_gu.copy()
        self.length_leaves = self.final_length_leaves.copy()
        self.length_inflos = self.final_length_inflos.copy()
        self.nb_leaf = np.where(
            self.radius_gu * 2 >= max_leafy_diameter_gu,
            0.,
            self.nb_internode
        ).astype(np.float32)
        self.any_is_growing = False

    @xs.runtime(args=('step'))
    def run_step(self, step):

        gu_growing = (self.gu_stage > 0.) & (self.gu_stage < self.nb_gu_stage) & (self.appeared == 1.)
        inflo_growing = (self.inflo_stage > 0.) & (self.inflo_stage < self.nb_inflo_stage) & (self.appeared == 1.)
        self.any_is_growing = np.any(gu_growing | inflo_growing)

        if np.any(self.appeared):
            self.nb_leaf[self.appeared == 1.] = self.nb_internode[self.appeared == 1.]

        params = self.parameters

        if np.any(gu_growing):

            radius_exponent_gu = params.radius_exponent_gu
            radius_coefficient_gu = params.radius_coefficient_gu

            self.radius_gu = radius_coefficient_gu * (self.nb_descendants + 1) ** radius_exponent_gu

            self.length_gu[gu_growing] = self.final_length_gu[gu_growing] / (1. + np.exp(
                -(self.gu_growth_tts[gu_growing] - params.t_ip_gu) / params.B_gu
            ))

            self.length_leaves[gu_growing] = self.get_length_leaves(
                self.final_length_leaves[gu_growing],
                self.leaf_growth_tts[gu_growing],
                params
            )

        if np.any(inflo_growing):

            radius_coefficient_inflo = params.radius_coefficient_inflo
            radius_slope_inflo = params.radius_slope_inflo

            self.radius_inflo[inflo_growing] = radius_coefficient_inflo + radius_slope_inflo * (self.inflo_stage[inflo_growing] / self.nb_inflo_stage)

            self.length_inflos[inflo_growing] = self.get_length_inflos(
                self.final_length_inflos[inflo_growing],
                self.inflo_growth_tts[inflo_growing],
                params
            )

        self.nb_leaf[self.radius_gu >= params.max_leafy_diameter_gu / 2.] = 0.
