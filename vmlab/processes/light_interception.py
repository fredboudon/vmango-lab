import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib

from . import environment, growth
from ._base.parameter import ParameterizedProcess


@xs.process
class LightInterception(ParameterizedProcess):
    """ Compute light interception for photosynthetically active radiation
    """

    sunlit_bs = None

    nb_gu = xs.global_ref('nb_gu')
    GR = xs.foreign(environment.Environment, 'GR')
    hour = xs.foreign(environment.Environment, 'hour')

    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')

    LA = xs.variable(
        dims=('GU'),
        intent='out',
        description='total leaf area per GU',
        attrs={
            'unit': 'm²'
        }
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

        self.sunlit_fractions_df = pd.read_csv(
            pathlib.Path(self.parameter_file_path).parent.joinpath(self.parameters.sunlit_fractions_file_path),
            sep='\\s+',
            usecols=['q10', 'q25', 'q50', 'q75', 'q90']
        )
        self.sunlit_bs = self.sunlit_fractions_df.iloc[:, 3].to_numpy(dtype=np.float32)
        self.LA = np.zeros(self.nb_gu, dtype=np.float32)
        self.LA_sunlit = np.zeros(self.nb_gu, dtype=np.float32)
        self.LA_shaded = np.zeros(self.nb_gu, dtype=np.float32)

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

        # leaf area (eq. 11) :
        self.LA = e_nleaf2LA_1 * self.nb_leaf ** e_nleaf2LA_2

        self.LA_sunlit = np.array([self.sunlit_bs * sunlit_ws * LA for LA in self.LA])
        self.LA_shaded = np.array([LA - LA_sunlit for LA, LA_sunlit in zip(self.LA, self.LA_sunlit)])

    def finalize_step(self):
        pass

    def finalize(self):
        pass
