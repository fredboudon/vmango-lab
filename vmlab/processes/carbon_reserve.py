import xsimlab as xs
import numpy as np

from . import (
    topology,
    growth,
    carbon_flow_coef
)
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonReserve(ParameterizedProcess):
    """Compute carbon available from reserves
    """

    carbon_allocation = xs.group_dict('carbon_allocation')
    appeared = xs.foreign(topology.Topology, 'appeared')
    month_begin_veg_cycle = xs.foreign(topology.Topology, 'month_begin_veg_cycle')

    nb_gu = xs.global_ref('nb_gu')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')
    radius_gu = xs.foreign(growth.Growth, 'radius_gu')
    length_gu = xs.foreign(growth.Growth, 'length_gu')

    is_photo_active = xs.foreign(carbon_flow_coef.CarbonFlowCoef, 'is_photo_active')

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
        description='carbon in stem "non-mobile" (not easily mobilized) reserves',
        attrs={
            'unit': 'g C'
        }
    )

    reserve_nmob_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='carbon in leaf "non-mobile" (not easily mobilized) reserves',
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

    def initialize(self):

        super(CarbonReserve, self).initialize()

        params = self.parameters

        cc_leaf = params.cc_leaf
        cc_stem = params.cc_stem
        r_DM_leaf_ini = params.r_DM_leaf_ini
        r_DM_stem_ini = params.r_DM_stem_ini
        r_mobile_leaf = params.r_mobile_leaf
        r_mobile_stem = params.r_mobile_stem
        r_storage_leaf_max = params.r_storage_leaf_max
        DM_stem_density = params.DM_stem_density
        DM_leaf_unit = params.DM_leaf_unit

        self.DM_structural_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * (1 - r_DM_leaf_ini), dtype=np.float32)
        self.reserve_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * r_DM_leaf_ini * cc_leaf, dtype=np.float32)

        self.DM_structural_stem = DM_stem_density * (1 - r_DM_stem_ini) * np.pi * 2. * self.radius_gu * self.length_gu
        self.reserve_stem = self.DM_structural_stem * r_DM_stem_ini * cc_stem

        self.reserve_mob = ((r_mobile_leaf * self.reserve_leaf) + (r_mobile_stem * self.reserve_stem)).astype(np.float32)

        self.reserve_nmob_leaf = (self.reserve_leaf * (1 - r_mobile_leaf)).astype(np.float32)
        self.reserve_nmob_stem = (self.reserve_stem * (1 - r_mobile_stem)).astype(np.float32)

        self.reserve_leaf_max = ((r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf * cc_leaf).astype(np.float32)

    @xs.runtime(args=('step_start'))
    def run_step(self, step_start):

        params = self.parameters
        cc_leaf = params.cc_leaf
        cc_stem = params.cc_stem
        r_DM_leaf_ini = params.r_DM_leaf_ini
        r_DM_stem_ini = params.r_DM_stem_ini
        r_mobile_leaf = params.r_mobile_leaf
        r_mobile_stem = params.r_mobile_stem
        r_storage_leaf_max = params.r_storage_leaf_max
        DM_stem_density = params.DM_stem_density
        DM_leaf_unit = params.DM_leaf_unit

        is_active = np.flatnonzero(self.is_photo_active == 1.)

        # reset reserve pools and update GU stem DM when a new veg. cycle begins
        if step_start.astype('datetime64[D]').item().month == self.month_begin_veg_cycle:
            self.DM_structural_stem = DM_stem_density * (1 - r_DM_stem_ini) * np.pi * 2. * self.radius_gu * self.length_gu
            self.reserve_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * r_DM_leaf_ini * cc_leaf, dtype=np.float32)
            self.reserve_stem = self.DM_structural_stem * r_DM_stem_ini * cc_stem
            self.reserve_mob = ((r_mobile_leaf * self.reserve_leaf) + (r_mobile_stem * self.reserve_stem)).astype(np.float32)
            self.reserve_nmob_leaf = (self.reserve_leaf * (1 - r_mobile_leaf)).astype(np.float32)
            self.reserve_nmob_stem = (self.reserve_stem * (1 - r_mobile_stem)).astype(np.float32)
            self.reserve_leaf_max = ((r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf * cc_leaf).astype(np.float32)

        if np.any(self.appeared):
            appeared = np.flatnonzero(self.appeared == 1.)
            self.DM_structural_leaf[appeared] = DM_leaf_unit * self.nb_leaf[appeared] * (1 - r_DM_leaf_ini)
            # GU growth/elongation is ignored during the cycle and will be updated at the start of the next veg. cycle
            self.DM_structural_stem[appeared] = DM_stem_density * (1 - r_DM_stem_ini) * np.pi * 2. * self.radius_gu[appeared] * self.length_gu[appeared]
            self.reserve_leaf[appeared] = DM_leaf_unit * self.nb_leaf[appeared] * r_DM_leaf_ini * cc_leaf
            self.reserve_stem[appeared] = self.DM_structural_stem[appeared] * r_DM_stem_ini * cc_stem
            self.reserve_mob[appeared] = (r_mobile_leaf * self.reserve_leaf[appeared]) + (r_mobile_stem * self.reserve_stem[appeared])
            self.reserve_nmob_leaf[appeared] = self.reserve_leaf[appeared] * (1 - r_mobile_leaf)
            self.reserve_nmob_stem[appeared] = self.reserve_stem[appeared] * (1 - r_mobile_stem)
            self.reserve_leaf_max[appeared] = (r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf[appeared] * cc_leaf

        assert not np.any(np.isnan(self.carbon_allocation[('carbon_allocation', 'reserve_leaf_delta')][is_active]))
        assert not np.any(np.isnan(self.carbon_allocation[('carbon_allocation', 'reserve_stem_delta')][is_active]))
        self.reserve_leaf[is_active] += self.carbon_allocation[('carbon_allocation', 'reserve_leaf_delta')][is_active]
        self.reserve_stem[is_active] += self.carbon_allocation[('carbon_allocation', 'reserve_stem_delta')][is_active]
        self.reserve_nmob_leaf[is_active] += self.carbon_allocation[('carbon_allocation', 'reserve_nmob_leaf_delta')][is_active]
        self.reserve_nmob_stem[is_active] += self.carbon_allocation[('carbon_allocation', 'reserve_nmob_stem_delta')][is_active]

        self.reserve_mob[is_active] = ((r_mobile_leaf * self.reserve_leaf[is_active]) + (r_mobile_stem * self.reserve_stem[is_active]))
