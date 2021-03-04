import xsimlab as xs
import numpy as np

from . import light_interception
from ._base.parameter import ParameterizedProcess


@xs.process
class Photosythesis(ParameterizedProcess):

    DM_fruit_max = xs.global_ref('DM_fruit_max')
    CU = xs.global_ref('CU')

    LA = xs.foreign(light_interception.LightInterception, 'LA')
    PAR = xs.foreign(light_interception.LightInterception, 'PAR')
    PAR_shaded = xs.foreign(light_interception.LightInterception, 'PAR_shaded')
    LA_sunlit = xs.foreign(light_interception.LightInterception, 'LA_sunlit')
    LA_shaded = xs.foreign(light_interception.LightInterception, 'LA_shaded')

    Pmax = xs.variable(
        dims=('CU'),
        intent='out',
        description='light-saturated leaf photosynthesis',
        attrs={
            'unit': 'µmol CO2 m-2 s-1'
        }
    )

    P_rate_sunlit = xs.variable(
        dims=('CU', 'hour'),
        intent='out',
        description='hourly photosynthetic rate per unit leaf area (for the hours of the day where PAR>0) for sunlit leaves',
        attrs={
            'unit': 'µmol CO2 m-2 s-1'
        }
    )

    P_rate_shaded = xs.variable(
        dims=('CU', 'hour'),
        intent='out',
        description='hourly photosynthetic rate per unit leaf area (for the hours of the day where PAR>0) for shaded leaves',
        attrs={
            'unit': 'µmol CO2 m-2 s-1'
        }
    )

    photo_sunlit = xs.variable(
        dims=('CU'),
        intent='out',
        description='carbon daily fixed by leaf photosynthesis of sunlit leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    photo_shaded = xs.variable(
        dims=('CU'),
        intent='out',
        description='carbon daily fixed by leaf photosynthesis of shaded leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    photo = xs.variable(
        dims=('CU'),
        intent='out',
        description='carbon daily fixed by leaf photosynthesis of sunlit and shaded leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    def initialize(self):

        super(Photosythesis, self).initialize()

        self.Pmax = np.zeros(self.CU.shape)
        self.P_rate_sunlit = np.zeros((self.CU.shape[0], 24))
        self.P_rate_shaded = np.zeros((self.CU.shape[0], 24))

        self.photo_shaded = np.zeros(self.CU.shape)
        self.photo_sunlit = np.zeros(self.CU.shape)
        self.photo = np.zeros(self.CU.shape)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        p_1 = params.p_1
        p_2 = params.p_2
        p_3 = params.p_3
        p_4 = params.p_4
        Pmax_max = params.Pmax_max
        k = params.k

        # light-saturated leaf photosynthesis (eq.1)
        self.Pmax = np.array([np.minimum((p_1 * (DM_fruit_max / LA) * p_2) / (p_1 * (DM_fruit_max / LA) + p_2), Pmax_max)
                              if LA > 0 else 0. for LA, DM_fruit_max in zip(self.LA, self.DM_fruit_max)])

        # photosynthetic rate per unit leaf area (eq.2)
        self.P_rate_sunlit = np.array([np.maximum(0., ((Pmax + p_3) * (1 - np.exp(-p_4 * self.PAR / (Pmax + p_3)))) - p_3) for Pmax in self.Pmax])
        self.P_rate_shaded = np.array([np.maximum(0., ((Pmax + p_3) * (1 - np.exp(-p_4 * self.PAR_shaded / (Pmax + p_3)))) - p_3) for Pmax in self.Pmax])

        # carbon assimilation by leaf photosynthesis (eq.3)
        self.photo_shaded = np.array([photo_shaded.sum() for photo_shaded in (self.P_rate_shaded * self.LA_shaded * k)])
        self.photo_sunlit = np.array([photo_sunlit.sum() for photo_sunlit in (self.P_rate_sunlit * self.LA_sunlit * k)])
        self.photo = self.photo_shaded + self.photo_sunlit

    def finalize_step(self):
        pass

    def finalize(self):
        pass
