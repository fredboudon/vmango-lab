import xsimlab as xs
import numpy as np

from . import (
    growth,
    light_interception,
    carbon_demand,
    carbon_allocation,
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class Photosythesis(ParameterizedProcess):
    """ Compute leaf photosythesis per GUs
        - photosythesis only for fully developed GUs (gu_stage >= 4.)
    """

    nb_gu = xs.global_ref('nb_gu')

    D_fruit = xs.foreign(carbon_demand.CarbonDemand, 'D_fruit')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')
    is_in_distance_to_fruit = xs.foreign(carbon_allocation.CarbonAllocation, 'is_in_distance_to_fruit')
    is_photo_active = xs.foreign(carbon_allocation.CarbonAllocation, 'is_photo_active')
    allocation_share = xs.foreign(carbon_allocation.CarbonAllocation, 'allocation_share')

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

    D_fruit_avg = xs.variable(
        dims=('GU'),
        intent='out',
        description='average daily carbon demand for fruit growth',
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
        self.D_fruit_avg = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=('step'))
    def run_step(self, step):

        params = self.parameters

        p_1 = params.p_1
        p_2 = params.p_2
        p_3 = params.p_3
        p_4 = params.p_4
        Pmax_max = params.Pmax_max
        Pmax_min = params.Pmax_min
        k = params.k

        if np.any(self.is_photo_active == 1.):

            is_active = np.flatnonzero(self.is_photo_active == 1.)
            is_fruiting = np.flatnonzero(self.nb_fruit > 0.)

            D_fruit_share = (self.is_in_distance_to_fruit * self.LA / np.vstack(np.nansum(self.is_in_distance_to_fruit * self.LA, axis=1))) * np.vstack(self.D_fruit[is_fruiting])
            # self.D_fruit_avg[np.nansum(D_fruit_share, axis=0) > 0.] = np.nanmean(D_fruit_share[:, np.nansum(D_fruit_share, axis=0) > 0.], axis=0)
            self.D_fruit_avg = np.nansum(D_fruit_share, axis=0)

            # light-saturated leaf photosynthesis (eq.1)
            self.Pmax[:] = 0.
            self.Pmax[is_active] = np.minimum(np.maximum((p_1 * (self.D_fruit_avg[is_active] / self.LA[is_active]) * p_2) / (p_1 * (self.D_fruit_avg[is_active] / self.LA[is_active]) + p_2), Pmax_min), Pmax_max)

            # photosynthetic rate per unit leaf area (eq.2)
            self.P_rate_sunlit[:] = 0.
            self.P_rate_shaded[:] = 0.
            self.P_rate_sunlit[is_active] = np.maximum(0., (np.vstack(self.Pmax + p_3) * (1 - np.exp(-p_4 * self.PAR / np.vstack(self.Pmax + p_3)))) - p_3)[is_active]
            self.P_rate_shaded[is_active] = np.maximum(0., (np.vstack(self.Pmax + p_3) * (1 - np.exp(-p_4 * self.PAR_shaded / np.vstack(self.Pmax + p_3)))) - p_3)[is_active]

            # carbon assimilation by leaf photosynthesis (eq.3)
            self.photo_shaded[:] = 0.
            self.photo_sunlit[:] = 0.
            self.photo[:] = 0.
            self.photo_shaded[is_active] = np.sum(self.P_rate_shaded[is_active] * self.LA_shaded[is_active] * k, axis=1)
            self.photo_sunlit[is_active] = np.sum(self.P_rate_sunlit[is_active] * self.LA_sunlit[is_active] * k, axis=1)
            self.photo[is_active] = self.photo_shaded[is_active] + self.photo_sunlit[is_active]
