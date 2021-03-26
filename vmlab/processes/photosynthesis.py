import xsimlab as xs
import numpy as np

from . import light_interception, carbon_demand
from ._base.parameter import ParameterizedProcess


@xs.process
class Photosythesis(ParameterizedProcess):
    """ Compute leaf photosythesis per GUs
        - photosythesis only for fully developed GUs (gu_stage >= 4.)
    """

    nb_gu = xs.global_ref('nb_gu')

    D_fruit = xs.foreign(carbon_demand.CarbonDemand, 'D_fruit')

    LA = xs.foreign(light_interception.LightInterception, 'LA')
    PAR = xs.foreign(light_interception.LightInterception, 'PAR')
    PAR_shaded = xs.foreign(light_interception.LightInterception, 'PAR_shaded')
    LA_sunlit = xs.foreign(light_interception.LightInterception, 'LA_sunlit')
    LA_shaded = xs.foreign(light_interception.LightInterception, 'LA_shaded')

    Pmax = xs.variable(
        dims=('GU'),
        intent='out',
        description='light-saturated leaf photosynthesis',
        attrs={
            'unit': 'µmol CO2 m-2 s-1'
        }
    )

    P_rate_sunlit = xs.variable(
        dims=('GU', 'hour'),
        intent='out',
        description='hourly photosynthetic rate per unit leaf area (for the hours of the day where PAR>0) for sunlit leaves',
        attrs={
            'unit': 'µmol CO2 m-2 s-1'
        }
    )

    P_rate_shaded = xs.variable(
        dims=('GU', 'hour'),
        intent='out',
        description='hourly photosynthetic rate per unit leaf area (for the hours of the day where PAR>0) for shaded leaves',
        attrs={
            'unit': 'µmol CO2 m-2 s-1'
        }
    )

    photo_sunlit = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon daily fixed by leaf photosynthesis of sunlit leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    photo_shaded = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon daily fixed by leaf photosynthesis of shaded leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    photo = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon daily fixed by leaf photosynthesis of sunlit and shaded leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    def initialize(self):

        super(Photosythesis, self).initialize()

        self.Pmax = np.zeros(self.nb_gu, dtype=np.float32)
        self.P_rate_sunlit = np.zeros((self.nb_gu, 24), dtype=np.float32)
        self.P_rate_shaded = np.zeros((self.nb_gu, 24), dtype=np.float32)

        self.photo_shaded = np.zeros(self.nb_gu, dtype=np.float32)
        self.photo_sunlit = np.zeros(self.nb_gu, dtype=np.float32)
        self.photo = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        p_1 = params.p_1
        p_2 = params.p_2
        p_3 = params.p_3
        p_4 = params.p_4
        Pmax_max = params.Pmax_max
        Pmax_min = params.Pmax_min
        k = params.k

        # light-saturated leaf photosynthesis (eq.1)
        self.Pmax = np.array([np.minimum(np.maximum((p_1 * (D_fruit / LA) * p_2) / (p_1 * (D_fruit / LA) + p_2), Pmax_min), Pmax_max)
                              if LA > 0 else 0. for LA, D_fruit in zip(self.LA, self.D_fruit)])

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
