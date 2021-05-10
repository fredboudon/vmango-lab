import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib

from . import (
    environment,
    growth
)
from ._base.parameter import ParameterizedProcess


@xs.process
class LightInterception(ParameterizedProcess):
    """ Compute light interception for photosynthetically active radiation
    """

    nb_gu = xs.global_ref('nb_gu')
    GR = xs.foreign(environment.Environment, 'GR')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')

    sunlit_fraction_df = xs.any_object()

    sunlit_fraction_col_default = xs.variable(
        static=True,
        default=4,
        description='The default column index from sunlit_fraction dataframe',
        attrs={
            'unit': '-'
        }
    )

    sunlit_fraction_col = xs.variable(
        dims='GU',
        intent='inout',
        description='The column index from sunlit_fraction dataframe',
        attrs={
            'unit': '-'
        },
        encoding={
            # set the fill value explicitly because when the array is grown the default is 0
            'fill_value': np.nan
        }
    )

    sunlit_fraction = xs.variable(
        dims=('GU', 'hour'),
        intent='out',
        description='Fraction of leaf area in direct sun light',
        attrs={
            'unit': 'm²/m²'
        }
    )

    LA = xs.variable(
        dims='GU',
        intent='out',
        description='total leaf area per GU',
        attrs={
            'unit': 'm²'
        }
    )

    PAR = xs.variable(
        dims='hour',
        intent='out',
        description='hourly photosynthetically active radiation',
        attrs={
            'unit': 'µmol photon m-2 s-1'
        }
    )

    PAR_shaded = xs.variable(
        dims='hour',
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

        self.sunlit_fraction_df = pd.read_csv(
            pathlib.Path(self.parameter_file_path).parent.joinpath(self.parameters.sunlit_fractions_file_path),
            sep='\\s+',
            usecols=['q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95']
        )

        self.sunlit_fraction_col[np.isnan(self.sunlit_fraction_col)] = self.sunlit_fraction_col_default
        self.sunlit_fraction = self.sunlit_fraction_df.iloc[:, self.sunlit_fraction_col].to_numpy(dtype=np.float32).T
        self.LA = np.zeros(self.nb_gu, dtype=np.float32)
        self.LA_sunlit = np.zeros(self.nb_gu, dtype=np.float32)
        self.LA_shaded = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):

        if np.any(np.isnan(self.sunlit_fraction_col)):
            # initialization of appearing GUs
            nan_sunlit_fractions = np.flatnonzero(np.isnan(self.sunlit_fraction_col))
            self.sunlit_fraction_col[nan_sunlit_fractions] = self.sunlit_fraction_col_default
            self.sunlit_fraction[nan_sunlit_fractions, :] = self.sunlit_fraction_df.iloc[:, self.sunlit_fraction_col[nan_sunlit_fractions]].to_numpy(dtype=np.float32).T

        params = self.parameters

        k_1 = params.k_1
        k_2 = params.k_2
        k_3 = params.k_3
        sunlit_ws = params.sunlit_ws
        e_nleaf2LA_1 = params.e_nleaf2LA_1
        e_nleaf2LA_2 = params.e_nleaf2LA_2

        # GR conversion form J/cm2/h to W/m2
        GR = self.GR / 3600 * 10000

        # photosynthetic active radiation (eq.10-19) :
        self.PAR = GR * k_1 * k_2
        self.PAR_shaded = k_3 * self.PAR

        # leaf area (eq. 11) :
        self.LA = e_nleaf2LA_1 * self.nb_leaf ** e_nleaf2LA_2

        self.LA_sunlit = self.sunlit_fraction * sunlit_ws * np.vstack(self.LA)
        self.LA_shaded = np.vstack(self.LA) - self.LA_sunlit
