import xsimlab as xs
import numpy as np

from . import environment, topology
from ._base.parameter import BaseParameterizedProcess


@xs.process
class Phenology(BaseParameterizedProcess):

    GU = xs.global_ref('GU')
    TM = xs.foreign(environment.Environment, 'TM')
    nb_inflo = xs.foreign(topology.Topology, 'nb_inflo')
    nb_leaf = xs.foreign(topology.Topology, 'nb_leaf')
    nb_fruit = xs.foreign(topology.Topology, 'nb_fruit')

    gu_stages = []
    inflo_stages = []

    leaf_growth_tts = xs.variable(
        dims='GU',
        intent='out'
    )
    leaf_pheno_tts = xs.variable(
        dims='GU',
        intent='out'
    )

    gu_growth_tts = xs.variable(
        dims='GU',
        intent='out'
    )
    gu_pheno_tts = xs.variable(
        dims='GU',
        intent='out'
    )
    gu_stage = xs.variable(
        dims='GU',
        intent='out',
        groups='phenology'
    )

    inflo_growth_tts = xs.variable(
        dims='GU',
        intent='out'
    )
    inflo_pheno_tts = xs.variable(
        dims='GU',
        intent='out'
    )
    inflo_stage = xs.variable(
        dims='GU',
        intent='out'
    )

    bloom_date = xs.variable(
        dims='GU',
        intent='inout',
        description='bloom date',
        attrs={
            'unit': 'date'
        },
        static=True
    )
    DAB = xs.variable(
        dims='GU',
        intent='out',
        description='days after bloom',
        attrs={
            'unit': 'd'
        }
    )
    dd_cum = xs.variable(
        dims='GU',
        intent='out',
        description='cumulated degree-days of the current day after bloom date',
        attrs={
            'unit': 'dd'
        }
    )
    dd_delta = xs.variable(
        dims='GU',
        intent='out',
        description='daily variation in degree days',
        attrs={
            'unit': 'dd day-1'
        }
    )

    def initialize(self):

        super(Phenology, self).initialize()

        params = self.parameters

        # apply t sums reversed so we do not visit a gu twice because it passed the stage threshold in the iteration
        params.Tbase_gu_stage = list(reversed(params.Tbase_gu_stage))
        params.Tthresh_gu_stage = list(reversed(params.Tthresh_gu_stage))
        self.gu_stages = list(reversed(range(len(params.Tbase_gu_stage))))
        params.Tbase_inflo_stage = list(reversed(params.Tbase_inflo_stage))
        params.Tthresh_inflo_stage = list(reversed(params.Tthresh_inflo_stage))
        self.inflo_stages = list(reversed(range(len(params.Tbase_inflo_stage))))

        self.leaf_growth_tts = np.zeros(self.GU.shape)
        self.leaf_pheno_tts = np.zeros(self.GU.shape)

        self.gu_growth_tts = np.zeros(self.GU.shape)
        self.gu_pheno_tts = np.zeros(self.GU.shape)
        self.gu_stage = np.zeros(self.GU.shape)

        self.inflo_growth_tts = np.zeros(self.GU.shape)
        self.inflo_pheno_tts = np.zeros(self.GU.shape)
        self.inflo_stage = np.zeros(self.GU.shape)

        self.bloom_date = np.array(self.bloom_date, dtype='datetime64[D]')
        self.DAB = np.zeros(self.GU.shape)
        self.dd_delta = np.zeros(self.GU.shape)
        self.dd_cum = np.zeros(self.GU.shape)

    @xs.runtime(args=('step_start'))
    def run_step(self, step_start):

        self.gu_pheno_tts[np.isnan(self.gu_pheno_tts)] = 0.
        self.gu_stage[np.isnan(self.gu_stage)] = 0.
        self.gu_growth_tts[np.isnan(self.gu_growth_tts)] = 0.
        self.leaf_growth_tts[np.isnan(self.gu_growth_tts)] = 0.

        params = self.parameters
        Tbase_leaf_growth = params.Tbase_leaf_growth

        # growth units

        Tbase_gu_growth = params.Tbase_gu_growth
        Tbase_gu_stage = params.Tbase_gu_stage
        Tthresh_gu_stage = params.Tthresh_gu_stage

        self.gu_growth_tts += max(0, self.TM - Tbase_gu_growth)

        # from max(gu_stages) to min(gu_stages)
        for stage, thresh, base in zip(self.gu_stages, Tthresh_gu_stage, Tbase_gu_stage):
            in_stage = (self.gu_stage >= stage) & (self.gu_stage < stage + 1)
            if not np.any(in_stage):
                continue
            self.gu_pheno_tts[in_stage] += max(0, self.TM - base)
            share = self.gu_pheno_tts[in_stage] / thresh
            self.gu_stage[in_stage] = np.where(share > 1., stage + 1., stage + share)
            self.gu_pheno_tts[np.nonzero(in_stage)] = np.where(share > 1., 0., self.gu_pheno_tts[np.nonzero(in_stage)])

        # inflorescences

        Tbase_inflo_stage = params.Tbase_inflo_stage
        Tthresh_inflo_stage = params.Tthresh_inflo_stage
        Tbase_inflo_growth = params.Tbase_inflo_growth

        has_inflos = (self.nb_inflo > 0.)

        self.inflo_growth_tts[~has_inflos] = 0.
        self.inflo_growth_tts[has_inflos] += max(0, self.TM - Tbase_inflo_growth)

        # from max(inflo_stages) to min(inflo_stages)
        self.inflo_stage[~has_inflos] = 0.
        self.inflo_pheno_tts[~has_inflos] = 0.
        for stage, thresh, base in zip(self.inflo_stages, Tthresh_inflo_stage, Tbase_inflo_stage):
            in_stage = has_inflos & (self.inflo_stage >= stage) & (self.inflo_stage < stage + 1)
            if not np.any(in_stage):
                continue
            self.inflo_pheno_tts[in_stage] += max(0, self.TM - base)
            share = self.inflo_pheno_tts[in_stage] / thresh
            self.inflo_stage[in_stage] = np.where(share > 1., stage + 1., stage + share)
            self.inflo_pheno_tts[np.nonzero(in_stage)] = np.where(share > 1., 0., self.inflo_pheno_tts[np.nonzero(in_stage)])

        # leaves

        Tbase_leaf_growth = params.Tbase_leaf_growth
        has_leaves = (self.nb_leaf > 0.)

        self.leaf_growth_tts[~has_leaves] = 0.
        self.leaf_growth_tts[has_leaves] += max(0, self.TM - Tbase_leaf_growth)

        # fruits

        Tbase_fruit_growth = params.Tbase_fruit_growth

        has_inflo_or_fruit = (step_start >= self.bloom_date) & ((self.nb_fruit > 0) | (self.nb_inflo > 0))

        self.DAB = np.where(
            has_inflo_or_fruit,
            (step_start - self.bloom_date).astype('timedelta64[D]') / np.timedelta64(1, 'D'),
            -1
        )
        self.dd_delta = np.where(
            has_inflo_or_fruit,
            max(0, self.TM - Tbase_fruit_growth),
            0.
        )
        self.dd_cum = np.where(
            has_inflo_or_fruit,
            self.dd_cum + self.dd_delta,
            0.
        )

    def finalize_step(self):
        pass

    def finalize(self):
        pass