import xsimlab as xs
import numpy as np
import warnings

from . import (
    environment,
    photosynthesis,
    light_interception,
    fruit_growth,
    topology,
    phenology
)
from ._base.parameter import BaseParameterizedProcess


@xs.process
class CarbonBalance(BaseParameterizedProcess):

    LFratio_previous = None

    TM_air = xs.foreign(environment.Environment, 'TM_air')
    T_fruit = xs.foreign(environment.Environment, 'T_fruit')
    GR = xs.foreign(environment.Environment, 'GR')

    photo = xs.foreign(photosynthesis.Photosythesis, 'photo')

    DM_fruit_max = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_max')
    DM_fruit_0 = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_0')
    D_fruit = xs.foreign(fruit_growth.FruitGrowth, 'D_fruit')

    dd_delta = xs.foreign(phenology.FlowerPhenology, 'dd_delta')

    GU = xs.foreign(topology.Topology, 'GU')

    LFratio = xs.foreign(light_interception.LightInterception, 'LFratio')

    DM_structural_stem = xs.variable(
        dims=('GU'),
        intent='out',
        description='dry mass of the structural part of stem',
        attrs={
            'unit': 'g DM'
        }
    )

    DM_structural_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='dry mass of the structural part of leaves',
        attrs={
            'unit': 'g DM'
        }
    )

    reserve_stem = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon in stem reserves',
        attrs={
            'unit': 'g C'
        }
    )

    reserve_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon in leaf reserves',
        attrs={
            'unit': 'g C'
        }
    )

    reserve_mob = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon in leaf and stem mobile reserves',
        attrs={
            'unit': 'g C'
        }
    )

    reserve_nmob_stem = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon in stem non-mobile reserves',
        attrs={
            'unit': 'g C'
        }
    )

    reserve_nmob_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon in leaf non-mobile reserves',
        attrs={
            'unit': 'g C'
        }
    )

    MR_stem = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of stem',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of leaves',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of fruits',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_repro = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of reproductive components (fruits)',
        attrs={
            'unit': 'g C day-1'
        }
    )

    MR_veget = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily maintenance respiration demand of vegetative components (leaves and stem)',
        attrs={
            'unit': 'g C day-1'
        }
    )

    assimilates = xs.variable(
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

    reserve_leaf_max = xs.variable(
        dims=('GU'),
        intent='out',
        description='maximal amount of carbon that can be stored in leaf reserves (threshold for reserve saturation)',
        attrs={
            'unit': 'g C'
        }
    )

    DM_fruit = xs.variable(
        dims=('GU'),
        intent='inout',
        description='fruit dry mass',
        attrs={
            'unit': 'g DM'
        },
        global_name='DM_fruit'
    )

    DM_fruit_delta = xs.variable(
        dims=('GU'),
        intent='out',
        description='change in fruit dry mass',
        attrs={
            'unit': 'g DM'
        }
    )

    def initialize(self):

        super(CarbonBalance, self).initialize()

        params = self.parameters

        cc_stem = params.cc_stem
        r_DM_stem_ini = params.r_DM_stem_ini
        DM_stem = params.DM_stem

        # initial amount of carbon in leaf and stem reserves :
        # self.reserve_leaf = np.ones(self.GU.shape) * (DM_leaf_unit * self.LFratio) * r_DM_leaf_ini * cc_leaf
        self.reserve_stem = np.ones(self.GU.shape) * DM_stem * r_DM_stem_ini * cc_stem

        self.DM_structural_stem = np.zeros(self.GU.shape)
        self.DM_structural_leaf = np.zeros(self.GU.shape)
        self.reserve_mob = np.zeros(self.GU.shape)
        self.reserve_nmob_stem = np.zeros(self.GU.shape)
        self.reserve_nmob_leaf = np.zeros(self.GU.shape)
        self.MR_stem = np.zeros(self.GU.shape)
        self.MR_leaf = np.zeros(self.GU.shape)
        self.MR_fruit = np.zeros(self.GU.shape)
        self.MR_repro = np.zeros(self.GU.shape)
        self.MR_veget = np.zeros(self.GU.shape)
        self.assimilates = np.zeros(self.GU.shape)
        self.required_DM_fruit = np.zeros(self.GU.shape)
        self.remains_1 = np.zeros(self.GU.shape)
        self.remains_2 = np.zeros(self.GU.shape)
        self.remains_3 = np.zeros(self.GU.shape)
        self.reserve_leaf_max = np.zeros(self.GU.shape)
        self.DM_fruit = np.zeros(self.GU.shape)
        self.DM_fruit_delta = np.zeros(self.GU.shape)
        # self.D_fruit = np.zeros(self.GU.shape)

        self.LFratio_previous = self.LFratio.copy()

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters

        r_DM_stem_ini = params.r_DM_stem_ini
        r_DM_leaf_ini = params.r_DM_leaf_ini
        DM_stem = params.DM_stem
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
        r_DM_leaf_ini = params.r_DM_leaf_ini
        cc_leaf = params.cc_leaf
        DM_leaf_unit = params.DM_leaf_unit
        # RGR_fruit_ini = params.RGR_fruit_ini

        # intitialize Fruit DM only if DM_fruit is still 0 and LFratio > 0
        self.DM_fruit = np.array([DM_fruit_0 if LFratio > 0 and DM_fruit == 0 else 0. if LFratio == 0 else DM_fruit
                                  for DM_fruit_0, DM_fruit, LFratio in zip(self.DM_fruit_0, self.DM_fruit, self.LFratio)])

        # initial amount of carbon in leaf and stem reserves : if LFratio is > 0
        if np.any(self.LFratio_previous == 0.):
            self.reserve_leaf = np.where(
                (self.LFratio_previous == 0.) & (self.LFratio > 0.),
                (DM_leaf_unit * self.LFratio) * r_DM_leaf_ini * cc_leaf,
                0
            )

        self.LFratio_previous = self.LFratio.copy()

        self.remains_1[:] = 0.
        self.remains_2[:] = 0.
        self.remains_3[:] = 0.

        # dry mass of stem and leaf structure
        self.DM_structural_stem = np.ones(self.GU.shape) * DM_stem * (1 - r_DM_stem_ini)
        self.DM_structural_leaf = DM_leaf_unit * self.LFratio * (1 - r_DM_leaf_ini)

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
        self.MR_stem = np.array([has_fruit * (MRR_stem * (Q10_stem ** ((self.TM_air - Tref) / 10)) * (DM_structural_stem + (reserve_stem / cc_stem)) / 24).sum()
                                for DM_structural_stem, reserve_stem, has_fruit in zip(self.DM_structural_stem, self.reserve_stem, self.LFratio > 0)])
        self.MR_fruit = np.array([(MRR_fruit * (Q10_fruit ** ((self.T_fruit - Tref) / 10)) * DM_fruit / 24).sum()
                                 for DM_fruit in self.DM_fruit])
        self.MR_leaf = np.array([has_fruit * ((self.GR > 0) * MRR_leaf * (Q10_leaf ** ((self.TM_air - Tref) / 10)) * (DM_structural_leaf + reserve_leaf / cc_leaf)).sum()
                                for DM_structural_leaf, reserve_leaf, has_fruit in zip(self.DM_structural_leaf, self.reserve_leaf, self.LFratio > 0)])

        # daily maintenance respiration for reproductive and vegetative components
        self.MR_repro = self.MR_fruit
        self.MR_veget = self.MR_stem + self.MR_leaf

        # CARBON ALLOCATION
        # Allocation of carbon according to organ demand and priority rules :
        #    1- maintenance of the system
        #    2- reproductive growth
        #    3- accumulation and replenishment of reserves in leaves and then in stem

        # pool of assimilates
        self.assimilates = self.photo + self.reserve_mob

        # 1- assimilates are used for maintenance respiration
        # Priority rules for maintenance respiration :
        #    1- vegetative components
        #    2- reproductive components

        assimilates_gt_mr_vegt = self.assimilates >= self.MR_veget
        mobilize_from_leaf = (self.assimilates + self.reserve_nmob_leaf >= self.MR_veget) & ~assimilates_gt_mr_vegt
        mobilize_from_stem = (self.assimilates + self.reserve_nmob_leaf + self.reserve_nmob_stem >= self.MR_veget) & ~assimilates_gt_mr_vegt & mobilize_from_leaf

        # print(assimilates_gt_mr_vegt, mobilize_from_leaf, mobilize_from_stem)
        # print(self.remains_1, self.photo, self.reserve_mob, self.MR_veget)

        # use of assimilates for maintenance respiration of vegetative components :
        self.remains_1 = np.where(
            assimilates_gt_mr_vegt,
            self.remains_1 + self.assimilates - self.MR_veget,
            0
        )

        # mobilization of non-mobile reserves if maintenance respiration is not satified by assimilates :
        # 1- mobilization of non-mobile reserves from leaves
        self.reserve_nmob_leaf = np.where(
            mobilize_from_leaf,
            self.assimilates + self.reserve_nmob_leaf - self.MR_veget,
            self.reserve_nmob_leaf
        )

        # 2- mobilization of non-mobile reserves from stem
        self.reserve_nmob_stem = np.where(
            mobilize_from_stem,
            self.assimilates + self.reserve_nmob_leaf + self.reserve_nmob_stem - self.MR_veget,
            self.reserve_nmob_stem
        )

        if not np.all(assimilates_gt_mr_vegt + mobilize_from_leaf + mobilize_from_stem):
            # TODO: What to do with variables?
            warnings.warn('Vegetative part of the system dies ...')

        # use of remaining assimilates for maintenance respiration of reproductive components :
        remaining_assimilates_lt_mr_repro = self.remains_1 < self.MR_repro

        # mobilization of fruit reserves if maintenance respiration is not satified by remaining assimilates :
        self.required_DM_fruit = np.where(
            remaining_assimilates_lt_mr_repro,
            (self.MR_repro - self.remains_1) / cc_fruit,
            0
        )
        mobilize_from_fruit = remaining_assimilates_lt_mr_repro & (self.required_DM_fruit < self.DM_fruit)
        self.DM_fruit = np.where(
            mobilize_from_fruit,
            self.DM_fruit - self.required_DM_fruit,
            self.DM_fruit
        )

        if not np.all(~remaining_assimilates_lt_mr_repro + mobilize_from_fruit):
            # TODO: What to do with variables?
            # death of reproductive components if maintenance respiration is not satisfied by remaining assimilates and fruit reserves :
            warnings.warn('Reproductive part of the system dies ...')

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

        reserve_max = (r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf * cc_leaf

        print(reserve_leaf_provi, reserve_max)

        self.reserve_leaf = np.where(
            reserve_leaf_provi > reserve_max,
            reserve_max,
            reserve_leaf_provi
        )

        self.reserve_stem = np.where(
            reserve_leaf_provi > reserve_max,
            reserve_stem_provi + reserve_leaf_provi - reserve_max,
            reserve_stem_provi
        )

        self.DM_fruit_delta = np.minimum(self.D_fruit, self.remains_2) / (cc_fruit + GRC_fruit)
        self.DM_fruit = self.DM_fruit + self.DM_fruit_delta

    def finalize_step(self):
        pass

    def finalize(self):
        pass
