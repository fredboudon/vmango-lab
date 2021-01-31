import xsimlab as xs
import numpy as np

from . import parameters
from . import environment
from . import growth_unit_growth
from . import phenology
from .base import BaseGrowthUnitProcess


@xs.process
class FruitGrowth(BaseGrowthUnitProcess):

    params = xs.foreign(parameters.Parameters, 'fruit_growth')

    TM = xs.foreign(environment.Environment, 'TM')

    GU = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'GU')

    bloom_date = xs.foreign(phenology.FlowerPhenology, 'bloom_date')
    dd_cum = xs.foreign(phenology.FlowerPhenology, 'dd_cum_gu')

    DM_fruit_0_gu = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit dry mass per fruit at the end of cell division (at 352.72 dd)',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    DM_fruit_max_gu = xs.variable(
        dims=('GU'),
        intent='out',
        description='potential maximal fruit dry mass per fruit (i.e. attained when fruit is grown under optimal conditions)',
        attrs={
            'unit': 'g DM'
        }
    )

    nb_fruits_ini = xs.variable(
        dims=('GU'),
        intent='inout',
        static=True
    )

    nb_fruits_gu = xs.variable(
        dims=('GU'),
        intent='out'
    )

    def initialize(self):

        _, params = self.params

        weight_1 = params.fruitDM0_weight_1
        mu_1 = params.fruitDM0_mu_1
        sigma_1 = params.fruitDM0_sigma_1
        weight_2 = params.fruitDM0_weight_2
        mu_2 = params.fruitDM0_mu_2
        sigma_2 = params.fruitDM0_sigma_2

        self.DM_fruit_max_gu = np.zeros(self.GU.shape)
        self.DM_fruit_0_gu = np.ones(self.GU.shape) * weight_1 * np.random.normal(mu_1, sigma_1) + weight_2 * np.random.normal(mu_2, sigma_2)

        self.nb_fruits_ini = np.array(self.nb_fruits_ini, dtype=np.int64)
        self.nb_fruits_gu = np.zeros(self.nb_fruits_ini.shape, dtype=np.int64)

    def step(self, nsteps, step, step_start, step_end, step_delta):

        _, params = self.params

        e_fruitDM02max_1 = params.e_fruitDM02max_1
        e_fruitDM02max_2 = params.e_fruitDM02max_2

        dd_cum_0 = params.dd_cum_0

        # carbon demand for fruit growth (eq.5-6-7)
        self.DM_fruit_max_gu = np.where(
            self.nb_fruits_gu > 0,
            e_fruitDM02max_1 * self.DM_fruit_0_gu ** e_fruitDM02max_2,
            0
        )

        self.nb_fruits_gu = np.where(
            self.dd_cum >= dd_cum_0,
            self.nb_fruits_ini,
            0.
        )

    def finalize_step(self):
        pass

    def finalize(self):
        pass
