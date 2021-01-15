import xsimlab as xs
import numpy as np

from . import parameters
from . import architecture
from . import environment


@xs.process
class FruitGrowth():

    params = xs.foreign(parameters.Parameters, 'fruit_growth')
    nb_fruits = xs.foreign(environment.Environment, 'nb_fruits')
    GU = xs.foreign(architecture.Architecture, 'GU')

    DM_fruit_0 = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit dry mass at the end of cell division (at 352.72 dd)',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    DM_fruit_max = xs.variable(
        dims=('GU'),
        intent='out',
        description='potential maximal fruit dry mass (i.e. attained when fruit is grown under optimal conditions)',
        attrs={
            'unit': 'g DM'
        }
    )

    def initialize(self):

        _, params = self.params

        weight_1 = params.fruitDM0_weight_1
        mu_1 = params.fruitDM0_mu_1
        sigma_1 = params.fruitDM0_sigma_1
        weight_2 = params.fruitDM0_weight_2
        mu_2 = params.fruitDM0_mu_2
        sigma_2 = params.fruitDM0_sigma_2

        self.DM_fruit_0 = np.ones(self.GU.shape) * weight_1 * np.random.normal(mu_1, sigma_1) + weight_2 * np.random.normal(mu_2, sigma_2)

    @xs.runtime(args=('step'))
    def run_step(self, step):

        _, params = self.params

        e_fruitDM02max_1 = params.e_fruitDM02max_1
        e_fruitDM02max_2 = params.e_fruitDM02max_2

        # carbon demand for fruit growth (eq.5-6-7)
        self.DM_fruit_max = np.where(
            self.nb_fruits > 0,
            e_fruitDM02max_1 * self.DM_fruit_0 ** e_fruitDM02max_2,
            0
        )

    def finalize_step(self):
        pass

    def finalize(self):
        pass
