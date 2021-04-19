import xsimlab as xs
import numpy as np

from . import (
    topology,
    growth,
    carbon_allocation
)
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonReserve(ParameterizedProcess):
    """Compute carbon available from reserves
    """

    carbon_balance = xs.group_dict('carbon_balance')
    appeared = xs.foreign(topology.Topology, 'appeared')
    month_begin_veg_cycle = xs.foreign(topology.Topology, 'month_begin_veg_cycle')

    nb_gu = xs.global_ref('nb_gu')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')

    is_photo_active = xs.foreign(carbon_allocation.CarbonAllocation, 'is_photo_active')

    DM_structural_stem = xs.variable(
        dims=('GU'),
        intent='out',
        description='dry mass of the structural part of stem',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    DM_structural_leaf = xs.variable(
        dims=('GU'),
        intent='out',
        description='dry mass of the structural part of leaves',
        attrs={
            'unit': 'g DM'
        },
        static=True
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
        DM_stem_gu = params.DM_stem_gu
        DM_leaf_unit = params.DM_leaf_unit

        self.DM_structural_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * (1 - r_DM_leaf_ini), dtype=np.float32)
        self.reserve_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * r_DM_leaf_ini * cc_leaf, dtype=np.float32)

        self.DM_structural_stem = np.full(self.nb_gu, DM_stem_gu * (1 - r_DM_stem_ini), dtype=np.float32)
        self.reserve_stem = np.full(self.nb_gu, DM_stem_gu * r_DM_stem_ini * cc_stem, dtype=np.float32)

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
        DM_stem_gu = params.DM_stem_gu
        DM_leaf_unit = params.DM_leaf_unit

        is_active = np.flatnonzero(self.is_photo_active == 1.)

        if step_start.astype('datetime64[D]').item().month == self.month_begin_veg_cycle:
            self.DM_structural_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * (1 - r_DM_leaf_ini), dtype=np.float32)
            self.reserve_leaf = np.full(self.nb_gu, DM_leaf_unit * self.nb_leaf * r_DM_leaf_ini * cc_leaf, dtype=np.float32)
            self.DM_structural_stem = np.full(self.nb_gu, DM_stem_gu * (1 - r_DM_stem_ini), dtype=np.float32)
            self.reserve_stem = np.full(self.nb_gu, DM_stem_gu * r_DM_stem_ini * cc_stem, dtype=np.float32)
            self.reserve_mob = ((r_mobile_leaf * self.reserve_leaf) + (r_mobile_stem * self.reserve_stem)).astype(np.float32)
            self.reserve_nmob_leaf = (self.reserve_leaf * (1 - r_mobile_leaf)).astype(np.float32)
            self.reserve_nmob_stem = (self.reserve_stem * (1 - r_mobile_stem)).astype(np.float32)
            self.reserve_leaf_max = ((r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf * cc_leaf).astype(np.float32)

        if np.any(self.appeared):
            appeared = np.flatnonzero(self.appeared == 1.)
            self.DM_structural_leaf[appeared] = DM_leaf_unit * self.nb_leaf[appeared] * (1 - r_DM_leaf_ini)
            self.DM_structural_stem[appeared] = DM_stem_gu * (1 - r_DM_stem_ini)
            self.reserve_leaf[appeared] = DM_leaf_unit * self.nb_leaf[appeared] * r_DM_leaf_ini * cc_leaf
            self.reserve_stem[appeared] = DM_stem_gu * r_DM_stem_ini * cc_stem
            self.reserve_mob[appeared] = (r_mobile_leaf * self.reserve_leaf[appeared]) + (r_mobile_stem * self.reserve_stem[appeared])
            self.reserve_nmob_leaf[appeared] = self.reserve_leaf[appeared] * (1 - r_mobile_leaf)
            self.reserve_nmob_stem[appeared] = self.reserve_stem[appeared] * (1 - r_mobile_stem)
            self.reserve_leaf_max[appeared] = (r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf[appeared] * cc_leaf

        assert not np.any(np.isnan(self.carbon_balance[('carbon_balance', 'reserve_leaf_delta')][is_active]))
        assert not np.any(np.isnan(self.carbon_balance[('carbon_balance', 'reserve_stem_delta')][is_active]))
        self.reserve_leaf[is_active] += self.carbon_balance[('carbon_balance', 'reserve_leaf_delta')][is_active]
        self.reserve_stem[is_active] += self.carbon_balance[('carbon_balance', 'reserve_stem_delta')][is_active]
        self.reserve_nmob_leaf[is_active] += self.carbon_balance[('carbon_balance', 'reserve_nmob_leaf_delta')][is_active]
        self.reserve_nmob_stem[is_active] += self.carbon_balance[('carbon_balance', 'reserve_nmob_stem_delta')][is_active]

        self.reserve_mob[is_active] = ((r_mobile_leaf * self.reserve_leaf[is_active]) + (r_mobile_stem * self.reserve_stem[is_active]))
