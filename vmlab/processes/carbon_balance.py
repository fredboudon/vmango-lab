import xsimlab as xs
import numpy as np
import warnings

from . import (
    growth,
    photosynthesis,
    carbon_reserve,
    carbon_demand,
    topology,
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonBalance(ParameterizedProcess):
    """
        - only for fully developed GUs (gu_stage >= 4.)
    """

    photo = xs.foreign(photosynthesis.Photosythesis, 'photo')
    full_bloom_date = xs.foreign(phenology.Phenology, 'full_bloom_date')

    D_fruit = xs.foreign(carbon_demand.CarbonDemand, 'D_fruit')

    reserve_mob = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_mob')
    reserve_nmob_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_nmob_leaf')
    reserve_nmob_stem = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_nmob_stem')

    DM_fruit_0 = xs.foreign(carbon_demand.CarbonDemand, 'DM_fruit_0')

    GU = xs.foreign(topology.Topology, 'GU')
    nb_fruit = xs.foreign(topology.Topology, 'nb_fruit')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')

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

    reserve_mob_delta = xs.variable(
        dims=('GU'),
        intent='inout',
        description='change in carbon in leaf and stem mobile reserves',
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

        self.carbon_supply = np.zeros(self.GU.shape, dtype=np.float32)
        self.required_DM_fruit = np.zeros(self.GU.shape, dtype=np.float32)
        self.remains_1 = np.zeros(self.GU.shape, dtype=np.float32)
        self.remains_2 = np.zeros(self.GU.shape, dtype=np.float32)
        self.remains_3 = np.zeros(self.GU.shape, dtype=np.float32)
        self.DM_fruit = np.zeros(self.GU.shape, dtype=np.float32)
        self.DM_fruit_delta = np.zeros(self.GU.shape, dtype=np.float32)

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
        # RGR_fruit_ini = params.RGR_fruit_ini

        self.DM_fruit = np.where(
            self.full_bloom_date == step_start,
            self.DM_fruit_0,
            self.DM_fruit
        )

        # CARBON ALLOCATION
        # Allocation of carbon according to organ demand and priority rules :
        #    1- maintenance of the system
        #    2- reproductive growth
        #    3- accumulation and replenishment of reserves in leaves and then in stem

        # pool of assimilates
        self.carbon_supply = self.photo + self.reserve_mob

        # 1- assimilates are used for maintenance respiration
        # Priority rules for maintenance respiration :
        #    1- vegetative components
        #    2- reproductive components

        assimilates_gt_mr_vegt = self.carbon_supply >= self.MR_veget
        mobilize_from_leaf = (self.carbon_supply + self.reserve_nmob_leaf >= self.MR_veget) & ~assimilates_gt_mr_vegt
        mobilize_from_stem = (self.carbon_supply + self.reserve_nmob_leaf + self.reserve_nmob_stem >= self.MR_veget) & ~assimilates_gt_mr_vegt & ~mobilize_from_leaf
        gu_died = ~assimilates_gt_mr_vegt & ~mobilize_from_leaf & ~mobilize_from_stem

        # use of assimilates for maintenance respiration of vegetative components :
        self.remains_1 = np.where(
            assimilates_gt_mr_vegt,
            self.remains_1 + self.carbon_supply - self.MR_veget,
            0.
        )

        # mobilization of non-mobile reserves if maintenance respiration is not satified by assimilates :
        # 1- mobilization of non-mobile reserves from leaves
        self.reserve_nmob_leaf = np.where(
            mobilize_from_leaf,
            self.carbon_supply + self.reserve_nmob_leaf - self.MR_veget,
            self.reserve_nmob_leaf
        )

        # 2- mobilization of non-mobile reserves from stem
        self.reserve_nmob_stem = np.where(
            mobilize_from_stem,
            self.carbon_supply + self.reserve_nmob_leaf + self.reserve_nmob_stem - self.MR_veget,
            self.reserve_nmob_stem
        )

        if np.any(gu_died):
            # TODO: What to do with variables?
            warnings.warn('Vegetative part of the system dies ...')
            self.reserve_nmob_leaf = np.where(
                gu_died,
                0.,
                self.reserve_nmob_leaf
            )
            self.reserve_nmob_stem = np.where(
                gu_died,
                0.,
                self.reserve_nmob_stem
            )

        # use of remaining assimilates for maintenance respiration of reproductive components :
        remaining_assimilates_lt_mr_repro = self.remains_1 < self.MR_repro

        # mobilization of fruit reserves if maintenance respiration is not satified by remaining assimilates :
        self.required_DM_fruit = np.where(
            remaining_assimilates_lt_mr_repro,
            (self.MR_repro - self.remains_1) / cc_fruit,
            0
        )
        mobilize_from_fruit = remaining_assimilates_lt_mr_repro & (self.required_DM_fruit < self.DM_fruit * self.nb_fruit)
        self.DM_fruit = np.where(
            mobilize_from_fruit,
            self.DM_fruit - self.required_DM_fruit / self.nb_fruit,
            self.DM_fruit
        )

        fruit_died = ~remaining_assimilates_lt_mr_repro & ~mobilize_from_fruit

        if not np.any(fruit_died):
            # TODO: What to do with variables?
            # death of reproductive components if maintenance respiration is not satisfied by remaining assimilates and fruit reserves :
            warnings.warn('Reproductive part of the system dies ...')
            self.DM_fruit = np.where(
                fruit_died,
                0.,
                self.DM_fruit
            )

        self.remains_2 = np.maximum(0, self.remains_1 - self.MR_repro)

        # 2- remaining assimilates are used for fruit growth

        self.remains_3 = self.remains_2 - np.minimum(self.D_fruit, self.remains_2)

        # 3- remaining assimilates are accumulated as reserves in leaves and stem
        # Priority rules for reserve storage :
        #    1- replenishment of mobile reserves of the stem
        #    2- storage of all remaining assimilates in leaf reserves up to a maximum threshold
        #    3- storage of all remaining assimilates in stem reserves

        reserve_stem_provi = self.reserve_nmob_stem + np.minimum(self.remains_3, self.reserve_stem * r_mobile_stem)
        reserve_leaf_provi = self.reserve_nmob_leaf + np.maximum(0, self.remains_3 - self.reserve_stem * r_mobile_stem)

        reserve_leaf_max = (r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf * cc_leaf

        self.reserve_leaf = np.where(
            reserve_leaf_provi > reserve_leaf_max,
            reserve_leaf_max,
            reserve_leaf_provi
        )

        self.reserve_stem = np.where(
            reserve_leaf_provi > reserve_leaf_max,
            reserve_stem_provi + reserve_leaf_provi - reserve_leaf_max,
            reserve_stem_provi
        )

        self.DM_fruit_delta = np.minimum(self.D_fruit, self.remains_2) / (cc_fruit + GRC_fruit) / self.nb_fruit
        self.DM_fruit = self.DM_fruit + self.DM_fruit_delta

    def finalize_step(self):
        pass

    def finalize(self):
        pass
