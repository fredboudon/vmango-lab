import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib

from . import environment, topology, fruit_growth
from vmlab.processes import BaseParameterizedProcess


@xs.process
class LightInterception(BaseParameterizedProcess):

    sunlit_bs = None

    GR = xs.foreign(environment.Environment, 'GR')
    hour = xs.foreign(environment.Environment, 'hour')

    rng = xs.global_ref('rng')

    nb_fruits = xs.foreign(fruit_growth.FruitGrowth, 'nb_fruits')
    nb_leaves = xs.foreign(topology.Topology, 'nb_leaves')
    GU = xs.foreign(topology.Topology, 'GU')

    LA = xs.variable(
        dims=('GU'),
        intent='out',
        description='total leaf area per CU',
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

        super(LightInterception, self).initialize()

        self.LFratio = np.array([nb_leaves / nb_fruits if nb_fruits > 0 else 0.
                                 for nb_leaves, nb_fruits in zip(self.nb_leaves, self.nb_fruits)])
        self.sunlit_fractions_df = pd.read_csv(
            pathlib.Path(self.parameter_file_path).parent.joinpath(self.parameters.sunlit_fractions_file_path),
            sep='\\s+',
            usecols=['q10', 'q25', 'q50', 'q75', 'q90']
        )
        self.sunlit_bs = self.sunlit_fractions_df.iloc[:, self.rng.integers(0, 5)].to_numpy(dtype=np.float32)
        self.LA = np.zeros(self.GU.shape)
        self.LA_sunlit = np.zeros(self.GU.shape)
        self.LA_shaded = np.zeros(self.GU.shape)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        k_1 = params.k_1
        k_2 = params.k_2
        k_3 = params.k_3
        sunlit_ws = params.sunlit_ws
        e_nleaf2LA_1 = params.e_nleaf2LA_1
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
        self.LA = e_nleaf2LA_1 * self.LFratio ** e_nleaf2LA_2

        self.LA_sunlit = np.array([self.sunlit_bs * sunlit_ws * LA for LA in self.LA])
        self.LA_shaded = np.array([LA - LA_sunlit for LA, LA_sunlit in zip(self.LA, self.LA_sunlit)])

    def finalize_step(self):
        pass

    def finalize(self):
        pass
