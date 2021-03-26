import xsimlab as xs
import numpy as np
import openalea.plantgl.all as pgl
import math
import zarr

from . import topology
from ._base.parameter import BaseParameterizedProcess


@xs.process
class Appearance(BaseParameterizedProcess):

    rng = None

    GU = xs.global_ref('GU')

    nb_descendants = xs.foreign(topology.Topology, 'nb_descendants')
    is_apical = xs.foreign(topology.Topology, 'is_apical')
    parent_is_apical = xs.foreign(topology.Topology, 'parent_is_apical')
    nb_inflo = xs.foreign(topology.Topology, 'nb_inflo')
    appeared_topo = xs.foreign(topology.Topology, 'appeared')
    flowered = xs.foreign(topology.Topology, 'flowered')
    seed = xs.foreign(topology.Topology, 'seed')
    is_initially_terminal = xs.foreign(topology.Topology, 'is_initially_terminal')

    appeared = xs.variable(
        dims='GU',
        intent='out'
    )
    final_length_gu = xs.variable(
        dims=('GU'),
        intent='inout',
        groups='appearance'
    )
    nb_internode = xs.variable(
        dims=('GU'),
        intent='out',
        groups='appearance'
    )
    final_length_internodes = xs.variable(
        dims=('GU'),
        intent='out',
        groups='appearance',
        encoding={
            'object_codec': zarr.JSON()
        }
    )
    final_length_leaves = xs.variable(
        dims=('GU'),
        intent='out',
        groups='appearance',
        encoding={
            'object_codec': zarr.JSON()
        }
    )
    final_length_inflos = xs.variable(
        dims=('GU'),
        intent='out',
        groups='appearance',
        encoding={
            'object_codec': zarr.JSON()
        }
    )

    def get_final_length_gu(self, is_apical, parent_is_apical, rng, params):
        mu, sigma = params.gu_length_distrib[(is_apical, parent_is_apical)]
        gu_length = rng.normal(mu, sigma)
        while (gu_length < 5) or (gu_length > 25):
            gu_length = rng.normal(mu, sigma)
        return gu_length

    def get_nb_internode(self, is_apical, final_length_gu, params):
        ratio, intercept = params.leaf_nb_distrib[(is_apical, )]
        return max(round(intercept + ratio * final_length_gu), 1)

    def get_final_length_internodes(self, is_apical, final_length_gu, nb_internode, rng, params):
        LEPF = 0.  # length of space before the first leaf
        if is_apical == 1.:
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

    def get_final_length_inflos(self, nb_inflo, rng, params):
        final_length_inflos = []
        mu, sigma = params.inflo_length_distrib
        for _ in range(int(nb_inflo)):
            inflo_length = rng.normal(mu, sigma)
            while (inflo_length < 5) or (inflo_length > 44):
                inflo_length = rng.normal(mu, sigma)
            final_length_inflos.append(inflo_length)
        return final_length_inflos

    def get_final_length_leaves(self, is_apical, nb_internode, get_final_length_leaf, rng, params):
        mu, sigma = params.leaf_length_distrib[(is_apical,)]
        leaf_length = rng.normal(mu, sigma)
        while (leaf_length < 5) or (leaf_length > 34):
            leaf_length = rng.normal(mu, sigma)
        return [leaf_length * get_final_length_leaf(i / max(1., float(nb_internode - 1.))) for i in range(int(nb_internode))]

    def get_final_length_leaf():
        pass

    def initialize(self):

        super(Appearance, self).initialize()

        self.rng = np.random.default_rng(seed=self.seed)

        params = self.parameters

        params.gu_length_distrib = {tuple(idx): distrib for idx, distrib in params.gu_length_distrib}
        params.leaf_nb_distrib = {tuple(idx): distrib for idx, distrib in params.leaf_nb_distrib}
        params.leaf_length_distrib = {tuple(idx): distrib for idx, distrib in params.leaf_length_distrib}

        self.get_final_length_gu = np.vectorize(self.get_final_length_gu, excluded={'rng', 'params'})
        self.get_nb_internode = np.vectorize(self.get_nb_internode, excluded={'params'})

        # np.array of type list
        self.get_final_length_internodes = np.vectorize(self.get_final_length_internodes, otypes=[object], excluded={'rng', 'params'})
        self.get_final_length_leaves = np.vectorize(self.get_final_length_leaves, otypes=[object], excluded={'get_final_length_leaf', 'rng', 'params'})
        self.get_final_length_inflos = np.vectorize(self.get_final_length_inflos, otypes=[object], excluded={'rng', 'params'})

        self.get_final_length_leaf = pgl.QuantisedFunction(
            pgl.NurbsCurve2D(
                pgl.Point3Array([(0, 1, 1), (0.00149779, 1.00072, 1), (1, 0.995671, 1), (1, 0.400121, 1)])
            )
        )

        self.appeared = np.ones(self.GU.shape, dtype=np.float32)
        self.nb_internode = np.ones(self.GU.shape, dtype=np.float32)
        self.final_length_internodes = np.full(self.GU.shape, None, dtype=object)
        self.final_length_leaves = np.full(self.GU.shape, None, dtype=object)
        self.final_length_inflos = np.full(self.GU.shape, None, dtype=object)

        self.run_step(-1)

    @xs.runtime(args=('step'))
    def run_step(self, step):

        params = self.parameters
        appeared = (self.appeared_topo == 1.) if step >= 0 else np.full(self.GU.shape, True)
        flowered = self.flowered == 1.

        if np.any(appeared):

            # growth units

            self.final_length_gu[appeared] = self.get_final_length_gu(
                self.is_apical[appeared],
                self.parent_is_apical[appeared],
                self.rng, params
            )

            # internodes

            self.nb_internode[appeared] = self.get_nb_internode(
                self.is_apical[appeared],
                self.final_length_gu[appeared],
                params
            )

            self.final_length_internodes[appeared] = self.get_final_length_internodes(
                self.is_apical[appeared],
                self.final_length_gu[appeared],
                self.nb_internode[appeared],
                self.rng, params
            )

            # leaves

            self.final_length_leaves[appeared] = self.get_final_length_leaves(
                self.is_apical[appeared],
                self.nb_internode[appeared],
                self.get_final_length_leaf, self.rng, params
            )

        # inflorescences

        if np.any(flowered):

            self.final_length_inflos[flowered] = self.get_final_length_inflos(
                self.nb_inflo[flowered],
                self.rng, params
            )

        self.appeared[appeared] = 1.
