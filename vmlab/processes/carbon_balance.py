import xsimlab as xs
import numpy as np
import warnings

from . import (
    photosynthesis,
    carbon_allocation,
    carbon_reserve,
    carbon_demand,
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonBalance(ParameterizedProcess):
    """
        - only for fully developed GUs (gu_stage >= 4.)
    """

    nb_gu = xs.global_ref('nb_gu')

    photo = xs.foreign(photosynthesis.Photosythesis, 'photo')

    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')
    fruited = xs.foreign(phenology.Phenology, 'fruited')

    allocation_share = xs.foreign(carbon_allocation.CarbonAllocation, 'allocation_share')
    is_photo_active = xs.foreign(carbon_allocation.CarbonAllocation, 'is_photo_active')

    DM_fruit_0 = xs.foreign(carbon_demand.CarbonDemand, 'DM_fruit_0')
    D_fruit = xs.foreign(carbon_demand.CarbonDemand, 'D_fruit')
    MR_repro = xs.foreign(carbon_demand.CarbonDemand, 'MR_repro')
    MR_veget = xs.foreign(carbon_demand.CarbonDemand, 'MR_veget')

    reserve_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_leaf')
    reserve_stem = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_stem')
    reserve_mob = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_mob')
    reserve_nmob_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_nmob_leaf')
    reserve_nmob_stem = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_nmob_stem')
    DM_structural_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'DM_structural_leaf')

    carbon_supply = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon available as assimilates from leaf photosynthesis and mobile reserves',
        attrs={
            'unit': 'g C'
        }
    )

    required_DM_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit dry mass required for fruit maintenance respiration not satisfied by remaining assimilates',
        attrs={
            'unit': 'g DM'
        }
    )

    remains_1 = xs.variable(
        dims=('GU'),
        intent='out',
        description='assimilates remaining after maintenance respiration of vegetative component',
        attrs={
            'unit': 'g C'
        }
    )

    remains_2 = xs.variable(
        dims=('GU'),
        intent='out',
        description='assimilates remaining after maintenance respiration of vegetative and reproductive components',
        attrs={
            'unit': 'g C'
        }
    )

    remains_3 = xs.variable(
        dims=('GU'),
        intent='out',
        description='assimilates remaining after maintenance respiration of vegetative and reproductive components and fruit growth',
        attrs={
            'unit': 'g C'
        }
    )

    DM_fruit = xs.variable(
        dims=('GU'),
        intent='inout',
        description='fruit dry mass for average fruit of growth unit',
        attrs={
            'unit': 'g DM'
        },
        groups='carbon_balance'
    )

    DM_fruit_delta = xs.variable(
        dims=('GU'),
        intent='inout',
        description='change in fruit dry mass for average fruit of growth unit',
        attrs={
            'unit': 'g DM'
        },
        groups='carbon_balance'
    )

    reserve_stem_delta = xs.variable(
        dims=('GU'),
        intent='inout',
        description='change in carbon in stem reserves',
        attrs={
            'unit': 'g C'
        },
        groups='carbon_balance'
    )

    reserve_leaf_delta = xs.variable(
        dims=('GU'),
        intent='inout',
        description='change in carbon in leaf reserves',
        attrs={
            'unit': 'g C'
        },
        groups='carbon_balance'
    )

    reserve_nmob_stem_delta = xs.variable(
        dims=('GU'),
        intent='inout',
        description='change in carbon in stem "non-mobile" (not easily mobilized) reserves',
        attrs={
            'unit': 'g C'
        },
        groups='carbon_balance'
    )

    reserve_nmob_leaf_delta = xs.variable(
        dims=('GU'),
        intent='inout',
        description='change in carbon in leaf "non-mobile" (not easily mobilized) reserves',
        attrs={
            'unit': 'g C'
        },
        groups='carbon_balance'
    )

    def initialize(self):

        super(CarbonBalance, self).initialize()

        self.carbon_supply = np.zeros(self.nb_gu, dtype=np.float32)
        self.required_DM_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.remains_1 = np.zeros(self.nb_gu, dtype=np.float32)
        self.remains_2 = np.zeros(self.nb_gu, dtype=np.float32)
        self.remains_3 = np.zeros(self.nb_gu, dtype=np.float32)
        self.DM_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.DM_fruit_delta = np.zeros(self.nb_gu, dtype=np.float32)
        self.reserve_leaf_delta = np.zeros(self.nb_gu, dtype=np.float32)
        self.reserve_stem_delta = np.zeros(self.nb_gu, dtype=np.float32)
        self.reserve_nmob_leaf_delta = np.zeros(self.nb_gu, dtype=np.float32)
        self.reserve_nmob_stem_delta = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=('step_start'))
    def run_step(self, step_start):

        self.remains_1[:] = 0.
        self.remains_2[:] = 0.
        self.remains_3[:] = 0.

        params = self.parameters

        r_mobile_stem = params.r_mobile_stem
        cc_leaf = params.cc_leaf
        r_storage_leaf_max = params.r_storage_leaf_max
        cc_fruit = params.cc_fruit
        GRC_fruit = params.GRC_fruit

        self.DM_fruit[np.flatnonzero(self.fruited)] = self.DM_fruit_0
        self.DM_fruit[self.nb_fruit == 0.] = 0.
        self.reserve_leaf_delta[:] = 0.
        self.reserve_stem_delta[:] = 0.
        self.reserve_nmob_leaf_delta[:] = 0.
        self.reserve_nmob_stem_delta[:] = 0.

        if np.any(self.is_photo_active == 1.):

            reserve_leaf = self.reserve_leaf.copy()
            reserve_stem = self.reserve_stem.copy()
            reserve_nmob_leaf = self.reserve_nmob_leaf.copy()
            reserve_nmob_stem = self.reserve_nmob_stem.copy()

            active = np.flatnonzero(self.is_photo_active == 1.)
            fruiting = np.flatnonzero(self.nb_fruit > 0.)

            # CARBON ALLOCATION
            # Allocation of carbon according to organ demand and priority rules :
            #    1- maintenance of the system
            #    2- reproductive growth
            #    3- accumulation and replenishment of reserves in leaves and then in stem

            # pool of assimilates
            self.carbon_supply[active] = self.photo[active] + self.reserve_mob[active]

            # 1- assimilates are used for maintenance respiration
            # Priority rules for maintenance respiration :
            #    1- vegetative components
            #    2- reproductive components

            assimilates_gte_mr_vegt = self.carbon_supply[active] >= self.MR_veget[active]
            mobilize_from_leaf = ((self.carbon_supply[active] + self.reserve_nmob_leaf[active]) >= self.MR_veget[active]) & ~assimilates_gte_mr_vegt
            mobilize_from_stem = ((self.carbon_supply[active] + self.reserve_nmob_leaf[active] + self.reserve_nmob_stem[active]) >= self.MR_veget[active]) & ~assimilates_gte_mr_vegt & ~mobilize_from_leaf
            gu_died = ~assimilates_gte_mr_vegt & ~mobilize_from_leaf & ~mobilize_from_stem

            # # use of assimilates for maintenance respiration of vegetative components :
            self.remains_1[active] = np.where(
                assimilates_gte_mr_vegt,
                self.remains_1[active] + self.carbon_supply[active] - self.MR_veget[active],
                0.
            )

            # mobilization of non-mobile reserves if maintenance respiration is not satified by assimilates :
            # 1- mobilization of non-mobile reserves from leaves
            reserve_nmob_leaf[active] = np.where(
                mobilize_from_leaf,
                self.carbon_supply[active] + self.reserve_nmob_leaf[active] - self.MR_veget[active],
                self.reserve_nmob_leaf[active]
            )

            # 2- mobilization of non-mobile reserves from stem
            reserve_nmob_stem[active] = np.where(
                mobilize_from_stem,
                self.carbon_supply[active] + reserve_nmob_leaf[active] + self.reserve_nmob_stem[active] - self.MR_veget[active],
                self.reserve_nmob_stem[active]
            )

            if np.any(gu_died):
                # TODO: What to do with variables?
                warnings.warn('Vegetative part of the system dies ...')

            # use of remaining assimilates for maintenance respiration of reproductive components :
            remaining_assimilates_lt_mr_repro = np.dot(self.allocation_share, self.remains_1) < self.MR_repro[fruiting]
            # remaining_assimilates_lt_mr_repro = self.remains_1 < self.MR_repro

            # mobilization of fruit reserves if maintenance respiration is not satified by remaining assimilates :
            self.required_DM_fruit[fruiting] = np.where(
                remaining_assimilates_lt_mr_repro,
                (self.MR_repro[fruiting] - np.dot(self.allocation_share, self.remains_1)) / cc_fruit,
                0
            )
            mobilize_from_fruit = remaining_assimilates_lt_mr_repro & (self.required_DM_fruit[fruiting] < (self.DM_fruit[fruiting] * self.nb_fruit[fruiting]))
            self.DM_fruit[fruiting] = np.where(
                mobilize_from_fruit,
                self.DM_fruit[fruiting] - self.required_DM_fruit[fruiting] / self.nb_fruit[fruiting],
                self.DM_fruit[fruiting]
            )

            fruit_died = remaining_assimilates_lt_mr_repro & ~mobilize_from_fruit

            if np.any(fruit_died):
                # TODO: What to do with variables?
                # death of reproductive components if maintenance respiration is not satisfied by remaining assimilates and fruit reserves :
                warnings.warn('Reproductive part of the system dies ...')

            self.remains_2[active] = np.maximum(0, self.remains_1[active] - self.MR_repro[active])

            # 2- remaining assimilates are used for fruit growth

            self.remains_3[active] = self.remains_2[active] - np.minimum(self.D_fruit[active], self.remains_2[active])

            # 3- remaining assimilates are accumulated as reserves in leaves and stem
            # Priority rules for reserve storage :
            #    1- replenishment of mobile reserves of the stem
            #    2- storage of all remaining assimilates in leaf reserves up to a maximum threshold
            #    3- storage of all remaining assimilates in stem reserves

            reserve_stem_provi = reserve_nmob_stem[active] + np.minimum(self.remains_3[active], self.reserve_stem[active] * r_mobile_stem)
            reserve_leaf_provi = reserve_nmob_leaf[active] + np.maximum(0, self.remains_3[active] - self.reserve_stem[active] * r_mobile_stem)

            reserve_leaf_max = (r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf[active] * cc_leaf

            reserve_leaf[active] = np.where(
                reserve_leaf_provi > reserve_leaf_max,
                reserve_leaf_max,
                reserve_leaf_provi
            )

            reserve_stem[active] = np.where(
                reserve_leaf_provi > reserve_leaf_max,
                reserve_stem_provi + reserve_leaf_provi - reserve_leaf_max,
                reserve_stem_provi
            )

            self.reserve_leaf_delta[active] = reserve_leaf[active] - self.reserve_leaf[active]
            self.reserve_stem_delta[active] = reserve_stem[active] - self.reserve_stem[active]
            self.reserve_nmob_leaf_delta[active] = reserve_nmob_leaf[active] - self.reserve_nmob_leaf[active]
            self.reserve_nmob_stem_delta[active] = reserve_nmob_stem[active] - self.reserve_nmob_stem[active]

            self.DM_fruit_delta[fruiting] = np.minimum(self.D_fruit[fruiting], np.dot(self.allocation_share, self.remains_2)) / (cc_fruit + GRC_fruit) / self.nb_fruit[fruiting]
            self.DM_fruit[fruiting] = self.DM_fruit[fruiting] + self.DM_fruit_delta[fruiting]

    def finalize_step(self):
        pass

    def finalize(self):
        pass
