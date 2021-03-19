import xsimlab as xs
import numpy as np
import openalea.plantgl.all as pgl
import math
import zarr
import typing

from . import topology, phenology
from ._base.parameter import BaseParameterizedProcess
from vmlab.enums import Position


@xs.process
class Growth(BaseParameterizedProcess):

    GU = xs.global_ref('GU')
    rng = xs.global_ref('rng')

    nb_descendants = xs.foreign(topology.Topology, 'nb_descendants')
    position = xs.foreign(topology.Topology, 'position')
    position_parent = xs.foreign(topology.Topology, 'position_parent')
    nb_inflo = xs.foreign(topology.Topology, 'nb_inflo')
    nb_leaf = xs.foreign(topology.Topology, 'nb_leaf')
    nb_fruit = xs.foreign(topology.Topology, 'nb_fruit')

    gu_growth_tts = xs.foreign(phenology.Phenology, 'gu_growth_tts')
    leaf_growth_tts = xs.foreign(phenology.Phenology, 'leaf_growth_tts')
    inflo_growth_tts = xs.foreign(phenology.Phenology, 'inflo_growth_tts')

    radius_gu = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )
    length_gu = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )
    final_length_gu = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )
    nb_internode = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )
    final_length_internodes = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth',
        encoding={
            'object_codec': zarr.JSON()
        }
    )
    final_length_leaves = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth',
        encoding={
            'object_codec': zarr.JSON()
        }
    )
    length_leaves = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth',
        encoding={
            'object_codec': zarr.JSON()
        }
    )
    final_length_inflo = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )
    length_inflo = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='growth'
    )

    def get_final_length_gu(self, position, position_parent, rng, params):
        mu, sigma = params.gu_length_distrib[(position, position_parent)]
        gu_length = rng.normal(mu, sigma)
        while (gu_length < 5) or (gu_length > 25):
            gu_length = rng.normal(mu, sigma)
        return gu_length

    def get_nb_internode(self, position, final_length_gu, params):
        ratio, intercept = params.leaf_nb_distrib[(position, )]
        return max(round(intercept + ratio * final_length_gu), 1)

    def get_final_length_internodes(self, position, final_length_gu, nb_internode, rng, params):
        LEPF = 0.  # length of space before the first leaf
        if position == Position.APICAL:
            mu, sigma = (2.007, 0.763)
            LEPF = rng.gamma(mu, sigma)
            while (LEPF < 0) or (LEPF > 8):
                LEPF = rng.gamma(mu, sigma)
        else:
            # length of space before the first leaf depend of GU's length
            LEPF = final_length_gu * 0.38 + 0.88

        nb_internode = nb_internode - 1
        if nb_internode <= 1:
            return final_length_gu

        lengths = [math.exp(-2.64 * i / float(nb_internode - 1.)) for i in range(int(nb_internode))]
        scaling = final_length_gu / sum(lengths)

        return [LEPF] + [length * scaling for length in lengths]

    def get_final_length_leaves(self, position, nb_internode, get_leaf_length, rng, params):
        mu, sigma = params.leaf_length_distrib[(position, )]
        leaf_length = rng.normal(mu, sigma)
        while (leaf_length < 5) or (leaf_length > 34):
            leaf_length = rng.normal(mu, sigma)

        return [leaf_length * get_leaf_length(i / max(1., float(nb_internode - 1.))) for i in range(int(nb_internode))]

    def get_final_length_inflo(self, rng, params):
        mu, sigma = params.inflo_length_distrib
        inflo_length = rng.normal(mu, sigma)
        while (inflo_length < 5) or (inflo_length > 44):
            inflo_length = rng.normal(mu, sigma)
        return inflo_length

    def get_length_leaves(self, final_length_leaves: typing.List[float], leaf_growth_tts, params):
        final_length_leaves = np.array(final_length_leaves)
        max_growth_rate = -0.0188725 + 0.0147985 * final_length_leaves * 4
        B = final_length_leaves / max_growth_rate
        return (final_length_leaves / (1. + np.exp(-(leaf_growth_tts - params.t_ip_leaf) / B))).tolist()

    def initialize(self):

        super(Growth, self).initialize()

        params = self.parameters

        params.gu_length_distrib = {tuple(idx): distrib for idx, distrib in params.gu_length_distrib}
        params.leaf_nb_distrib = {tuple(idx): distrib for idx, distrib in params.leaf_nb_distrib}
        params.leaf_length_distrib = {tuple(idx): distrib for idx, distrib in params.leaf_length_distrib}
        params.t_ip_gu = self.rng.normal(params.t_ip_gu_mean, params.t_ip_gu_sd)

        self.get_final_length_gu = np.vectorize(self.get_final_length_gu, excluded={'rng', 'params'})
        self.get_nb_internode = np.vectorize(self.get_nb_internode, excluded={'params'})
        self.get_final_length_internodes = np.vectorize(self.get_final_length_internodes, otypes=[np.object], excluded={'rng', 'params'})
        self.get_final_length_leaves = np.vectorize(self.get_final_length_leaves, otypes=[np.object], excluded={'get_leaf_length', 'rng', 'params'})
        self.get_final_length_inflo = np.vectorize(self.get_final_length_inflo, excluded={'rng', 'params'})

        self.get_length_leaves = np.vectorize(self.get_length_leaves, otypes=[np.object], excluded={'leaf_growth_tts', 'params'})
        self.get_leaf_length = pgl.QuantisedFunction(
            pgl.NurbsCurve2D(
                pgl.Point3Array([(0, 1, 1), (0.00149779, 1.00072, 1), (1, 0.995671, 1), (1, 0.400121, 1)])
            )
        )

        self.final_length_gu = np.zeros(self.GU.shape)
        self.final_length_inflo = np.zeros(self.GU.shape)
        self.length_gu = np.zeros(self.GU.shape)
        # self.final_length_internodes = np.full(self.GU.shape, None)
        # self.final_length_leaves = np.full(self.GU.shape, None)
        # self.length_leaves = np.full(self.GU.shape, None)

    @xs.runtime(args=())
    def run_step(self):

        self.radius_gu[np.isnan(self.radius_gu)] = 0.
        self.nb_internode[np.isnan(self.nb_internode)] = 0.
        self.length_gu[np.isnan(self.length_gu)] = 0.
        self.final_length_gu[np.isnan(self.final_length_gu)] = 0.

        params = self.parameters

        # growth units

        radius_exponent_gu = params.radius_exponent_gu
        radius_coefficient_gu = params.radius_coefficient_gu

        self.radius_gu = radius_coefficient_gu * (self.nb_descendants + 1) ** radius_exponent_gu

        no_final_length_gu = (self.final_length_gu == 0)
        if np.any(no_final_length_gu):
            self.final_length_gu[no_final_length_gu] = self.get_final_length_gu(
                self.position[no_final_length_gu],
                self.position_parent[no_final_length_gu],
                self.rng, params
            )

        self.length_gu = self.final_length_gu / (1. + np.exp(-(self.gu_growth_tts - params.t_ip_gu) / params.B_gu))

        # internodes

        no_nb_internode = (self.nb_internode == 0)
        if np.any(no_nb_internode):
            self.nb_internode[no_nb_internode] = self.get_nb_internode(
                self.position[no_nb_internode],
                self.final_length_gu[no_nb_internode],
                params
            )

        no_final_length_internodes = (np.equal(self.final_length_internodes, None)) | (self.final_length_internodes == 0)
        if np.any(no_final_length_internodes):
            self.final_length_internodes[no_final_length_internodes] = self.get_final_length_internodes(
                self.position[no_final_length_internodes],
                self.final_length_gu[no_final_length_internodes],
                self.nb_internode[no_final_length_internodes],
                self.rng, params
            )

        # leaves

        no_final_length_leaves = (np.equal(self.final_length_leaves, None)) | (self.final_length_leaves == 0)
        if np.any(no_final_length_leaves):
            self.final_length_leaves[no_final_length_leaves] = self.get_final_length_leaves(
                self.position[no_final_length_leaves],
                self.nb_internode[no_final_length_leaves],
                self.get_leaf_length, self.rng, params
            )

        # fix <string>:49: RuntimeWarning: invalid value encountered in double_scalars
        self.length_leaves = self.get_length_leaves(
            self.final_length_leaves,
            self.leaf_growth_tts,
            params
        )

        # inflorescences

        no_final_length_inflo = (self.final_length_inflo == 0)
        if np.any(no_final_length_inflo):
            self.final_length_inflo[no_final_length_inflo] = self.get_final_length_inflo(
                self.rng, params
            )

        self.length_inflo = self.final_length_inflo / (1 + np.exp(-(self.inflo_growth_tts - params.t_ip_inflo) / params.B_inflo))

    def finalize_step(self):
        pass

    def finalize(self):
        pass
