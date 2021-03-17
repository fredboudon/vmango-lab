import xsimlab as xs
import numpy as np

from . import environment
from . import topology
from . import phenology
from ._base.parameter import BaseParameterizedProcess


@xs.process
class CarbonDemand(BaseParameterizedProcess):
    """Compute carbon demand for all organs and processes:
        - actual maintenance (stem, leaf, fruit)
        - potential growth (fruit)
        - maintenance only for fully developed GUs (gu_stage >= 4.)

        TODO:
            - add growth & maintenance demand for GU (gu_stage < 4.)
    """

    TM = xs.foreign(environment.Environment, 'TM')

    GU = xs.foreign(topology.Topology, 'GU')

    bloom_date = xs.foreign(phenology.Phenology, 'bloom_date')
    dd_cum = xs.foreign(phenology.Phenology, 'dd_cum')
    dd_delta = xs.foreign(phenology.Phenology, 'dd_delta')
    DM_fruit = xs.global_ref('DM_fruit')

    rng = xs.global_ref('rng')

    DM_fruit_0 = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit dry mass per fruit at the end of cell division (at 352.72 dd)',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    DM_fruit_max = xs.variable(
        dims=('GU'),
        intent='out',
        description='potential maximal fruit dry mass per fruit (i.e. attained when fruit is grown under optimal conditions)',
        attrs={
            'unit': 'g DM'
        }
    )

    D_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='total daily carbon demand for fruit growth of all fruits of gu',
        attrs={
            'unit': 'g C day-1'
        }
    )

    nb_fruits_ini = xs.variable(
        dims=('GU'),
        intent='inout',
        static=True
    )

    nb_fruits = xs.variable(
        dims=('GU'),
        intent='out'
    )

    def initialize(self):

        super(CarbonDemand, self).initialize()

        params = self.parameters

        weight_1 = params.fruitDM0_weight_1
        mu_1 = params.fruitDM0_mu_1
        sigma_1 = params.fruitDM0_sigma_1
        weight_2 = params.fruitDM0_weight_2
        mu_2 = params.fruitDM0_mu_2
        sigma_2 = params.fruitDM0_sigma_2

        self.DM_fruit_max = np.zeros(self.GU.shape)
        self.DM_fruit_0 = np.ones(self.GU.shape) * weight_1 * self.rng.normal(mu_1, sigma_1) + weight_2 * self.rng.normal(mu_2, sigma_2)
        self.D_fruit = np.zeros(self.GU.shape)

        self.nb_fruits_ini = np.array(self.nb_fruits_ini)
        self.nb_fruits = np.zeros(self.nb_fruits_ini.shape)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        e_fruitDM02max_1 = params.e_fruitDM02max_1
        e_fruitDM02max_2 = params.e_fruitDM02max_2
        dd_cum_0 = params.dd_cum_0
        RGR_fruit_ini = params.RGR_fruit_ini
        cc_fruit = params.cc_fruit
        GRC_fruit = params.GRC_fruit

        # set nb_fruits to nb_fruits_ini if dd_cum_0 is reached
        self.nb_fruits = np.where(
            self.dd_cum >= dd_cum_0,
            self.nb_fruits_ini,
            0.
        )

        # carbon demand for fruit growth (eq.5-6-7)
        self.DM_fruit_max = np.where(
            self.nb_fruits > 0,
            e_fruitDM02max_1 * self.DM_fruit_0 ** e_fruitDM02max_2,
            0
        )
        self.D_fruit = np.array([dd_delta * (cc_fruit + GRC_fruit) * RGR_fruit_ini * DM_fruit * (1 - (DM_fruit / DM_fruit_max)) * nb_fruits
                                 if DM_fruit_max > 0 else 0. for dd_delta, DM_fruit, DM_fruit_max, nb_fruits in zip(self.dd_delta, self.DM_fruit, self.DM_fruit_max, self.nb_fruits)])

    def finalize_step(self):
        pass

    def finalize(self):
        pass
