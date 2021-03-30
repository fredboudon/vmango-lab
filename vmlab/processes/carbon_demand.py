import xsimlab as xs
import numpy as np

from . import (
    environment,
    carbon_allocation,
    carbon_reserve,
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonDemand(ParameterizedProcess):
    """Compute carbon demand for all organs and processes:
        - actual maintenance (stem, leaf, fruit)
        - potential growth (fruit)
        - maintenance only for fully developed GUs (gu_stage >= 4.)

        TODO:
            - add growth & maintenance demand for GU (gu_stage < 4.)
    """

    rng = None

    seed = xs.global_ref('seed')
    nb_gu = xs.global_ref('nb_gu')

    is_photo_active = xs.foreign(carbon_allocation.CarbonAllocation, 'is_photo_active')

    carbon_balance = xs.group_dict('carbon_balance')

    TM = xs.foreign(environment.Environment, 'TM')
    TM_day = xs.foreign(environment.Environment, 'TM_day')
    GR = xs.foreign(environment.Environment, 'GR')

    fruit_growth_tts_delta = xs.foreign(phenology.Phenology, 'fruit_growth_tts_delta')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')

    DM_structural_stem = xs.foreign(carbon_reserve.CarbonReserve, 'DM_structural_stem')
    DM_structural_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'DM_structural_leaf')
    reserve_stem = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_stem')
    reserve_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_leaf')

    MR_stem = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of stem',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of fruits',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_repro = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of reproductive components (fruits)',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_veget = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of vegetative components (leaves and stem)',
        attrs={
            'unit': 'g C day-1'
        }
    )

    DM_fruit_0 = xs.variable(
        intent='out',
        description='fruit dry mass per fruit at the end of cell division (at 352.72 dd)',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    DM_fruit_max = xs.variable(
        intent='out',
        description='potential maximal fruit dry mass per fruit (i.e. attained when fruit is grown under optimal conditions)',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    D_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='total daily carbon demand for fruit growth of all fruits of gu',
        attrs={
            'unit': 'g C day-1'
        }
    )

    def initialize(self):

        super(CarbonDemand, self).initialize()

        self.rng = np.random.default_rng(seed=self.seed)

        params = self.parameters

        weight_1 = params.fruitDM0_weight_1
        mu_1 = params.fruitDM0_mu_1
        sigma_1 = params.fruitDM0_sigma_1
        weight_2 = params.fruitDM0_weight_2
        mu_2 = params.fruitDM0_mu_2
        sigma_2 = params.fruitDM0_sigma_2
        e_fruitDM02max_1 = params.e_fruitDM02max_1
        e_fruitDM02max_2 = params.e_fruitDM02max_2

        self.DM_fruit_0 = np.float32(weight_1 * self.rng.normal(mu_1, sigma_1) + weight_2 * self.rng.normal(mu_2, sigma_2))
        self.DM_fruit_max = np.float32(e_fruitDM02max_1 * self.DM_fruit_0 ** e_fruitDM02max_2)

        self.D_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_stem = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_leaf = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_repro = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_veget = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        params = self.parameters

        RGR_fruit_ini = params.RGR_fruit_ini
        cc_fruit = params.cc_fruit
        cc_stem = params.cc_stem
        cc_leaf = params.cc_leaf
        GRC_fruit = params.GRC_fruit
        MRR_stem = params.MRR_stem
        MRR_leaf = params.MRR_leaf
        MRR_fruit = params.MRR_fruit
        Q10_stem = params.Q10_stem
        Q10_leaf = params.Q10_leaf
        Q10_fruit = params.Q10_fruit
        Tref = params.Tref

        is_active = np.flatnonzero(self.is_photo_active == 1.)
        has_fruit = np.flatnonzero(self.nb_fruit > 0)

        self.MR_stem[:] = 0.
        self.MR_fruit[:] = 0.
        self.MR_leaf[:] = 0.
        self.MR_repro[:] = 0.
        self.MR_veget[:] = 0.

        if np.any(self.is_photo_active == 1.):

            DM_fruit = np.where(
                (self.nb_fruit > 0.) & (self.carbon_balance[('carbon_balance', 'DM_fruit')] == 0.),
                self.DM_fruit_0,
                self.carbon_balance[('carbon_balance', 'DM_fruit')]
            )

            # carbon demand for fruit growth (eq.5-6-7) :
            # maybe problem if Tbase_fruit_growth <=  self.TM_day at day of full_bloom_date
            self.D_fruit = self.fruit_growth_tts_delta * (cc_fruit + GRC_fruit) * RGR_fruit_ini * DM_fruit * (1 - (DM_fruit / self.DM_fruit_max)) * self.nb_fruit

            # MAINTENANCE RESPIRATION (eq.4)
            # daily maintenance respiration for the stem, leaves (only during dark hours) and fruits
            self.MR_stem[is_active] = np.sum(MRR_stem / 24 * (Q10_stem ** ((self.TM - Tref) / 10.)) * np.vstack(self.DM_structural_stem[is_active] + (self.reserve_stem[is_active] / cc_stem)), axis=1)
            self.MR_leaf[is_active] = np.sum((self.GR > 0.) * MRR_leaf * (Q10_leaf ** ((self.TM - Tref) / 10)) * np.vstack(self.DM_structural_leaf[is_active] + (self.reserve_leaf[is_active] / cc_leaf)), axis=1)
            self.MR_fruit[has_fruit] = np.sum(MRR_fruit / 24 * (Q10_fruit ** ((self.TM - Tref) / 10.)) * np.vstack(DM_fruit[has_fruit] * self.nb_fruit[has_fruit]), axis=1)

            # daily maintenance respiration for reproductive and vegetative components
            self.MR_repro[has_fruit] = self.MR_fruit[has_fruit]
            self.MR_veget[is_active] = self.MR_stem[is_active] + self.MR_leaf[is_active]

    def finalize_step(self):
        pass

    def finalize(self):
        pass
