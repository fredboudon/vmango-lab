import xsimlab as xs
import numpy as np

from . import (
    environment,
    topology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class Phenology(ParameterizedProcess):

    GU = xs.foreign(topology.Topology, 'GU')
    archdev = xs.group_dict('arch_dev')
    TM_day = xs.foreign(environment.Environment, 'TM_day')
    harvest = xs.group_dict('harvest')

    gu_stages = []
    inflo_stages = []

    leaf_growth_tts = xs.variable(
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
    nb_gu_stage = xs.variable(
        intent='out',
        groups='phenology',
        static=True
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
        intent='out',
        groups='phenology'
    )
    nb_inflo_stage = xs.variable(
        intent='out',
        groups='phenology',
        static=True
    )
    full_bloom_date = xs.variable(
        dims='GU',
        intent='out',
        description='full bloom date',
        attrs={
            'unit': 'date'
        },
        static=True
    )
    DAFB = xs.variable(
        dims='GU',
        intent='out',
        description='days after full bloom',
        attrs={
            'unit': 'days'
        }
    )
    fruit_growth_tts = xs.variable(
        dims='GU',
        intent='out',
        description='cumulated degree-days of the current day after full bloom date',
        attrs={
            'unit': 'degree-days'
        }
    )
    fruit_growth_tts_delta = xs.variable(
        dims='GU',
        intent='out',
        description='daily change in degree-days',
        attrs={
            'unit': 'degree-days day-1'
        }
    )
    fruited = xs.variable(
        dims='GU',
        intent='out',
        description='fruit appeared today',
        attrs={
            'unit': '-'
        }
    )
    flowered = xs.variable(dims='GU', intent='out', groups='phenology')
    nb_inflo = xs.variable(dims='GU', intent='out', groups='phenology')
    nb_fruit = xs.variable(dims='GU', intent='out', groups='phenology')

    def initialize(self):

        super(Phenology, self).initialize()

        params = self.parameters

        self.nb_gu_stage = len(params.Tbase_gu_stage)
        self.nb_inflo_stage = len(params.Tbase_inflo_stage)

        self.gu_stages = list(reversed(range(self.nb_gu_stage)))
        self.inflo_stages = list(reversed(range(self.nb_inflo_stage)))

        # apply t sums reversed so we do not visit a gu twice because it passed the stage threshold in the iteration
        params.Tbase_gu_stage = list(reversed(params.Tbase_gu_stage))
        params.Tthresh_gu_stage = list(reversed(params.Tthresh_gu_stage))
        params.Tbase_inflo_stage = list(reversed(params.Tbase_inflo_stage))
        params.Tthresh_inflo_stage = list(reversed(params.Tthresh_inflo_stage))

        self.leaf_growth_tts = np.zeros(self.GU.shape, dtype=np.float32)

        self.gu_growth_tts = np.zeros(self.GU.shape, dtype=np.float32)
        self.gu_pheno_tts = np.zeros(self.GU.shape, dtype=np.float32)
        self.gu_stage = np.full(self.GU.shape, np.float32(self.nb_gu_stage))

        self.inflo_growth_tts = np.zeros(self.GU.shape, dtype=np.float32)
        self.inflo_pheno_tts = np.zeros(self.GU.shape, dtype=np.float32)
        self.inflo_stage = np.zeros(self.GU.shape)

        self.full_bloom_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.DAFB = np.zeros(self.GU.shape, dtype=np.float32)
        self.fruit_growth_tts = np.zeros(self.GU.shape, dtype=np.float32)
        self.fruit_growth_tts_delta = np.zeros(self.GU.shape, dtype=np.float32)

        self.nb_inflo = np.zeros(self.GU.shape, dtype=np.float32)
        self.nb_fruit = np.zeros(self.GU.shape, dtype=np.float32)
        self.flowered = np.zeros(self.GU.shape, dtype=np.float32)
        self.fruited = np.zeros(self.GU.shape, dtype=np.float32)

    @xs.runtime(args=('step_start'))
    def run_step(self, step_start):

        self.gu_pheno_tts[np.isnan(self.gu_pheno_tts)] = 0.
        self.gu_stage[np.isnan(self.gu_stage)] = 0.
        self.gu_growth_tts[np.isnan(self.gu_growth_tts)] = 0.
        self.leaf_growth_tts[np.isnan(self.gu_growth_tts)] = 0.
        self.flowered[:] = 0.
        self.fruited[:] = 0.

        self.flowered[self.archdev[('arch_dev', 'pot_flowering_date')] == step_start] = 1.
        self.nb_inflo[self.flowered == 1.] = self.archdev[('arch_dev', 'pot_nb_inflo')][self.flowered == 1.]

        params = self.parameters
        Tbase_leaf_growth = params.Tbase_leaf_growth

        # growth units

        Tbase_gu_growth = params.Tbase_gu_growth
        Tbase_gu_stage = params.Tbase_gu_stage
        Tthresh_gu_stage = params.Tthresh_gu_stage

        self.gu_growth_tts[self.gu_stage < 4.] += max(0, self.TM_day - Tbase_gu_growth)

        # from max(gu_stages) to min(gu_stages)
        for stage, thresh, base in zip(self.gu_stages, Tthresh_gu_stage, Tbase_gu_stage):
            in_stage = (self.gu_stage >= stage) & (self.gu_stage < stage + 1)
            if not np.any(in_stage):
                continue
            self.gu_pheno_tts[in_stage] += max(0, self.TM_day - base)
            share = self.gu_pheno_tts[in_stage] / thresh
            self.gu_stage[in_stage] = np.where(share > 1., stage + 1., stage + share)
            self.gu_pheno_tts[np.nonzero(in_stage)] = np.where(share > 1., 0., self.gu_pheno_tts[np.nonzero(in_stage)])

        # inflorescences

        Tbase_inflo_stage = params.Tbase_inflo_stage
        Tthresh_inflo_stage = params.Tthresh_inflo_stage
        Tbase_inflo_growth = params.Tbase_inflo_growth

        has_inflo = (self.nb_inflo > 0.)

        self.inflo_growth_tts[~has_inflo] = 0.
        self.inflo_growth_tts[has_inflo & (self.inflo_stage < 5.)] += max(0, self.TM_day - Tbase_inflo_growth)

        # from max(inflo_stages) to min(inflo_stages)
        self.inflo_stage[~has_inflo] = 0.
        self.inflo_pheno_tts[~has_inflo] = 0.
        for stage, thresh, base in zip(self.inflo_stages, Tthresh_inflo_stage, Tbase_inflo_stage):
            in_stage = has_inflo & (self.inflo_stage >= stage) & (self.inflo_stage < stage + 1)
            if not np.any(in_stage):
                continue
            self.inflo_pheno_tts[in_stage] += max(0, self.TM_day - base)
            share = self.inflo_pheno_tts[in_stage] / thresh
            self.inflo_stage[in_stage] = np.where(share > 1., stage + 1., stage + share)
            self.inflo_pheno_tts[np.nonzero(in_stage)] = np.where(share > 1., 0., self.inflo_pheno_tts[np.nonzero(in_stage)])

        # leaves

        Tbase_leaf_growth = params.Tbase_leaf_growth

        self.leaf_growth_tts[self.gu_stage < self.nb_gu_stage] += max(0, self.TM_day - Tbase_leaf_growth)

        # fruits

        if np.any((self.inflo_stage < self.nb_inflo_stage) | (self.nb_fruit > 0.)):

            Tbase_fruit_growth = params.Tbase_fruit_growth
            Tthresh_fruit_stage = params.Tthresh_fruit_stage

            self.full_bloom_date = np.where(
                has_inflo & np.isnat(self.full_bloom_date) & (self.inflo_stage >= 2.),
                step_start,
                self.full_bloom_date
            )

            self.DAFB = np.where(
                has_inflo & ~np.isnat(self.full_bloom_date) & (self.harvest[('harvest', 'ripeness_index')] < 1.) & (self.archdev[('arch_dev', 'pot_nb_fruit')] > 0),
                (step_start - self.full_bloom_date).astype('timedelta64[D]') / np.timedelta64(1, 'D'),
                0.
            ).astype(np.float32)

            self.fruit_growth_tts_delta = np.where(
                (self.DAFB > 0) & (self.harvest[('harvest', 'ripeness_index')] < 1.),
                max(0, self.TM_day - Tbase_fruit_growth),
                0.
            ).astype(np.float32)

            self.fruit_growth_tts = np.where(
                (self.DAFB > 0) & (self.harvest[('harvest', 'ripeness_index')] < 1.),
                self.fruit_growth_tts + self.fruit_growth_tts_delta,
                0.
            ).astype(np.float32)

            self.fruited[
                (self.fruit_growth_tts >= Tthresh_fruit_stage) &
                (self.nb_fruit == 0) &
                (self.archdev[('arch_dev', 'pot_nb_fruit')] > 0)
            ] = 1.

            self.nb_fruit[np.flatnonzero(self.fruited)] = self.archdev[('arch_dev', 'pot_nb_fruit')][
                np.flatnonzero(self.fruited)
            ]

            self.nb_fruit[np.flatnonzero(self.harvest[('harvest', 'harvested')])] = 0.
