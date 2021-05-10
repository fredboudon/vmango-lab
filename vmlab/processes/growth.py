import xsimlab as xs
import numpy as np
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

    harvest = xs.group_dict('harvest')

    gu_growth_tts = xs.foreign(phenology.Phenology, 'gu_growth_tts')
    leaf_growth_tts = xs.foreign(phenology.Phenology, 'leaf_growth_tts')
    inflo_growth_tts = xs.foreign(phenology.Phenology, 'inflo_growth_tts')
    gu_stage = xs.foreign(phenology.Phenology, 'gu_stage')
    inflo_stage = xs.foreign(phenology.Phenology, 'inflo_stage')
    nb_gu_stage = xs.foreign(phenology.Phenology, 'nb_gu_stage')
    nb_inflo_stage = xs.foreign(phenology.Phenology, 'nb_inflo_stage')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')

    final_length_gu = xs.foreign(appearance.Appearance, 'final_length_gu')
    nb_internode = xs.foreign(appearance.Appearance, 'nb_internode')
    final_length_leaves = xs.foreign(appearance.Appearance, 'final_length_leaves')
    final_length_inflos = xs.foreign(appearance.Appearance, 'final_length_inflos')
    appeared = xs.foreign(appearance.Appearance, 'appeared')

    any_is_growing = xs.variable(intent='out', groups='growth')

    leaf_senescence_enabled = xs.variable(static=True, default=True)

    radius_gu = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth',
        attrs={
            'unit': 'cm'
        }
    )

    length_gu = xs.variable(
        dims=('GU'),
        intent='out',
        groups='growth',
        attrs={
            'unit': 'cm'
        }
    )

    length_leaves = xs.any_object(
        groups='growth'
    )

    length_inflos = xs.any_object(
        groups='growth'
    )

    radius_inflo = xs.variable(
        dims=('GU'),
        intent='out',
        groups='growth'
    )

    nb_leaf = xs.variable(
        dims='GU',
        intent='inout',
        groups=('growth', 'growth_inout'),
        encoding={
            'fill_value': np.nan
        }
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

        radius_gu_isnan = np.isnan(self.radius_gu)
        self.radius_gu[radius_gu_isnan] = (radius_coefficient_gu * (self.nb_descendants[radius_gu_isnan] + 1) ** radius_exponent_gu).astype(np.float32)
        self.radius_inflo = np.zeros(self.GU.shape, dtype=np.float32)
        self.length_gu = self.final_length_gu.copy()
        self.length_leaves = self.final_length_leaves.copy()
        self.length_inflos = self.final_length_inflos.copy()

        self.nb_leaf[np.isnan(self.nb_leaf)] = self.nb_internode[np.isnan(self.nb_leaf)]
        if self.leaf_senescence_enabled:
            self.nb_leaf[self.radius_gu * 2. >= max_leafy_diameter_gu] = 0.

        self.any_is_growing = False

    @xs.runtime(args=('step'))
    def run_step(self, step):

        if self.length_leaves.shape != self.GU.shape:
            length_leaves = np.empty(self.GU.shape, dtype=object)
            length_leaves[0:self.length_leaves.shape[0]] = self.length_leaves
            length_leaves[self.length_leaves.shape[0]:] = None
            self.length_leaves = length_leaves
            length_inflos = np.empty(self.GU.shape, dtype=object)
            length_inflos[0:self.length_inflos.shape[0]] = self.length_inflos
            length_inflos[self.length_inflos.shape[0]:] = None
            self.length_inflos = length_inflos

        gu_growing = (self.gu_stage > 0.) & (self.gu_stage < self.nb_gu_stage) & (self.appeared == 1.)
        inflo_growing = (self.inflo_stage > 0.) & (self.inflo_stage < self.nb_inflo_stage) & (self.appeared == 1.)
        fruit_growing = (self.harvest[('harvest', 'ripeness_index')] < 1.) & (self.nb_fruit > 0.)
        self.any_is_growing = np.any(gu_growing | inflo_growing | fruit_growing)

        if np.any(self.appeared):
            is_uninitialized = (self.appeared == 1.) & np.isnan(self.nb_leaf)
            self.nb_leaf[is_uninitialized] = self.nb_internode[is_uninitialized]

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

        if self.leaf_senescence_enabled:
            self.nb_leaf[self.radius_gu * 2. >= params.max_leafy_diameter_gu] = 0.
