import xsimlab as xs
import numpy as np

from . import (
    environment,
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

    TM_air = xs.foreign(environment.Environment, 'TM_air')
    T_fruit = xs.foreign(environment.Environment, 'T_fruit')

    fruit_growth_tts_delta = xs.foreign(phenology.Phenology, 'fruit_growth_tts_delta')

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

        self.DM_fruit_0 = weight_1 * self.rng.normal(mu_1, sigma_1) + weight_2 * self.rng.normal(mu_2, sigma_2)
        self.DM_fruit_max = e_fruitDM02max_1 * self.DM_fruit_0 ** e_fruitDM02max_2

        self.D_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_stem = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_leaf = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_repro = np.zeros(self.nb_gu, dtype=np.float32)
        self.MR_veget = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        RGR_fruit_ini = params.RGR_fruit_ini
        cc_fruit = params.cc_fruit
        GRC_fruit = params.GRC_fruit

        self.D_fruit = np.array([fruit_growth_tts_delta * (cc_fruit + GRC_fruit) * RGR_fruit_ini * DM_fruit * (1 - (DM_fruit / DM_fruit_max)) * nb_fruits
                                 if DM_fruit_max > 0 else 0. for fruit_growth_tts_delta, DM_fruit, DM_fruit_max, nb_fruits in zip(self.fruit_growth_tts_delta, self.DM_fruit, self.DM_fruit_max, self.nb_fruits)])

    def finalize_step(self):
        pass

    def finalize(self):
        pass
