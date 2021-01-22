import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib
import random

from . import growth_unit_growth
from . import fruit_growth
from . import environment
from . import parameters


@xs.process
class LightInterception():

    sunlit_bs = None

    params = xs.foreign(parameters.Parameters, 'light_interception')

    GR = xs.foreign(environment.Environment, 'GR')
    hour = xs.foreign(environment.Environment, 'hour')

    nb_fruits = xs.foreign(fruit_growth.FruitGrowth, 'nb_fruits')

    GU = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'GU')
    nb_leaves = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'nb_leaves')

    LA = xs.variable(
        dims=('GU'),
        intent='out',
        description='total leaf area per branch',
        attrs={
            'unit': 'm²'
        }
    )

    LFratio = xs.variable(
        dims=('GU'),
        intent='out'
    )

    PAR = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly photosynthetically active radiation',
        attrs={
            'unit': 'µmol photon m-2 s-1'
        }
    )

    PAR_shaded = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly photosynthetically active radiation received by shaded leaves',
        attrs={
            'unit': 'µmol photon m-2 s-1'
        }
    )

    LA_sunlit = xs.variable(
        dims=('GU', 'hour'),
        intent='out',
        description='hourly leaf area of sunlit leaves (for the hours of the day where PAR>0)',
        attrs={
            'unit': 'm²'
        }
    )

    LA_shaded = xs.variable(
        dims=('GU', 'hour'),
        intent='out',
        description='hourly leaf area of shaded leaves (for the hours of the day where PAR>0)',
        attrs={
            'unit': 'm²'
        }
    )

    def initialize(self):
        self.LFratio = np.array([nb_leaves / nb_fruits if nb_fruits > 0 else 0.
                                 for nb_leaves, nb_fruits in zip(self.nb_leaves, self.nb_fruits)])
        self.sunlit_fractions_df = pd.read_csv(
            pathlib.Path(self.params[0].parent).joinpath(self.params[1].sunlit_fractions_file_path),
            sep='\\s+',
            usecols=['q10', 'q25', 'q50', 'q75', 'q90']
        )
        self.sunlit_bs = self.sunlit_fractions_df.iloc[:, random.randrange(0, 5)].to_numpy(dtype=np.float32)

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        _, params = self.params

        k_1 = params.k_1
        k_2 = params.k_2
        k_3 = params.k_3
        sunlit_ws = params.sunlit_ws
        e_nleaf2LA_2 = params.e_nleaf2LA_2

        # hour = pd.Timestamp(step_start).hour

        # GR conversion form J/cm2/h to W/m2
        GR = self.GR / 3600 * 10000

        # photosynthetic active radiation (eq.10-19) :
        self.PAR = GR * k_1 * k_2
        self.PAR_shaded = k_3 * self.PAR

        self.LFratio = np.array([nb_leaves / nb_fruits if nb_fruits > 0 else 0.
                                 for nb_leaves, nb_fruits in zip(self.nb_leaves, self.nb_fruits)])

        # leaf area (eq. 11) :
        self.LA = self.params[1].e_nleaf2LA_1 * self.LFratio ** e_nleaf2LA_2

        self.LA_sunlit = np.array([self.sunlit_bs * sunlit_ws * LA for LA in self.LA])
        self.LA_shaded = np.array([LA - LA_sunlit for LA, LA_sunlit in zip(self.LA, self.LA_sunlit)])

    def finalize_step(self):
        pass

    def finalize(self):
        pass
