import xsimlab as xs
import numpy as np
import warnings

from . import (
    growth,
    photosynthesis,
    carbon_reserve,
    carbon_demand,
    topology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonBalance(ParameterizedProcess):
    """
        - only for fully developed GUs (gu_stage >= 4.)
    """

    photo = xs.foreign(photosynthesis.Photosythesis, 'photo')

    D_fruit = xs.foreign(carbon_demand.CarbonDemand, 'D_fruit')

    reserve_mob = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_mob')
    reserve_nmob_leaf = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_nmob_leaf')
    reserve_nmob_stem = xs.foreign(carbon_reserve.CarbonReserve, 'reserve_nmob_stem')

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
        global_name='DM_fruit'
    )

    DM_fruit_delta = xs.variable(
        dims=('GU'),
        intent='out',
        description='change in fruit dry mass for average fruit of growth unit',
        attrs={
            'unit': 'g DM'
        }
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

        self.carbon_supply = np.zeros(self.GU.shape)
        self.required_DM_fruit = np.zeros(self.GU.shape)
        self.remains_1 = np.zeros(self.GU.shape)
        self.remains_2 = np.zeros(self.GU.shape)
        self.remains_3 = np.zeros(self.GU.shape)
        self.DM_fruit = np.zeros(self.GU.shape)
        self.DM_fruit_delta = np.zeros(self.GU.shape)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        r_DM_stem_ini = params.r_DM_stem_ini
        r_DM_leaf_ini = params.r_DM_leaf_ini
        DM_stem = params.DM_stem_gu
        DM_leaf_unit = params.DM_leaf_unit
        r_mobile_leaf = params.r_mobile_leaf
        r_mobile_stem = params.r_mobile_stem
        MRR_stem = params.MRR_stem
        MRR_leaf = params.MRR_leaf
        MRR_fruit = params.MRR_fruit
        Q10_stem = params.Q10_stem
        Q10_leaf = params.Q10_leaf
        Q10_fruit = params.Q10_fruit
        Tref = params.Tref
        cc_stem = params.cc_stem
        cc_leaf = params.cc_leaf
        r_storage_leaf_max = params.r_storage_leaf_max
        cc_fruit = params.cc_fruit
        GRC_fruit = params.GRC_fruit
        # RGR_fruit_ini = params.RGR_fruit_ini

        # intitialize Fruit DM only if DM_fruit is still 0 and LFratio > 0
        # use actual nb_fruit (not pot. nb_fruit)
        self.DM_fruit = np.array([DM_fruit_0 if LFratio > 0 and DM_fruit == 0 else 0. if LFratio == 0 else DM_fruit
                                  for DM_fruit_0, DM_fruit, LFratio in zip(self.DM_fruit_0, self.DM_fruit, self.LFratio)])

        self.remains_1 = np.zeros(self.GU.shape)
        self.remains_2 = np.zeros(self.GU.shape)
        self.remains_3 = np.zeros(self.GU.shape)

        # dry mass of stem and leaf structure
        self.DM_structural_stem = np.ones(self.GU.shape) * DM_stem * (1 - r_DM_stem_ini)
        self.DM_structural_leaf = DM_leaf_unit * self.nb_leaf * (1 - r_DM_leaf_ini)

        # carbon demand for fruit growth (eq.5-6-7) :
        # self.D_fruit = np.array([dd_delta * (cc_fruit + GRC_fruit) * RGR_fruit_ini * DM_fruit * (1 - (DM_fruit / DM_fruit_max))
        #                          if DM_fruit_max > 0 else 0. for dd_delta, DM_fruit, DM_fruit_max in zip(self.dd_delta, self.DM_fruit, self.DM_fruit_max)])

        # CARBON AVAILABLE FROM RESERVE MOBILIZATION

        # mobile amount of reserves (eq.8-9)
        self.reserve_mob = (r_mobile_leaf * self.reserve_leaf) + (r_mobile_stem * self.reserve_stem)

        # non-mobile amount of reserves
        self.reserve_nmob_leaf = self.reserve_leaf * (1 - r_mobile_leaf)
        self.reserve_nmob_stem = self.reserve_stem * (1 - r_mobile_stem)

        # MAINTENANCE RESPIRATION (eq.4)

        # daily maintenance respiration for the stem, leaves (only during dark hours) and fruits
        self.MR_stem = np.sum((MRR_stem * (Q10_stem ** ((self.TM_air - Tref) / 10)) * (self.DM_structural_stem + (self.reserve_stem / cc_stem)) / 24))
        self.MR_leaf = np.sum((self.GR > 0) * MRR_leaf * (Q10_leaf ** ((self.TM_air - Tref) / 10)) * (self.DM_structural_leaf + self.reserve_leaf / cc_leaf))
        self.MR_fruit = np.array([(MRR_fruit * (Q10_fruit ** ((self.T_fruit - Tref) / 10)) * DM_fruit / 24).sum() * nb_fruit
                                 for DM_fruit, nb_fruit in zip(self.DM_fruit, self.nb_fruit)])

        # daily maintenance respiration for reproductive and vegetative components
        self.MR_repro = self.MR_fruit
        self.MR_veget = self.MR_stem + self.MR_leaf

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
