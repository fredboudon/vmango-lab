import xsimlab as xs
import numpy as np

from . import growth
from ._base.parameter import ParameterizedProcess


@xs.process
class CarbonReserve(ParameterizedProcess):
    """Compute carbon available from reserves
    """

    carbon_balance = xs.group_dict('carbon_balance')

    nb_gu = xs.global_ref('nb_gu')
    nb_leaf = xs.foreign(growth.Growth, 'nb_leaf')

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

        self.DM_structural_leaf = DM_leaf_unit * self.nb_leaf * (1 - r_DM_leaf_ini)
        self.reserve_leaf = DM_leaf_unit * self.nb_leaf * r_DM_leaf_ini * cc_leaf

        self.DM_structural_stem = np.full(self.nb_gu, DM_stem_gu * (1 - r_DM_stem_ini))
        self.reserve_stem = np.full(self.nb_gu, DM_stem_gu * r_DM_stem_ini * cc_stem)

        self.reserve_mob = (r_mobile_leaf * self.reserve_leaf) + (r_mobile_stem * self.reserve_stem)

        self.reserve_nmob_leaf = self.reserve_leaf * (1 - r_mobile_leaf)
        self.reserve_nmob_stem = self.reserve_stem * (1 - r_mobile_stem)

        self.reserve_leaf_max = (r_storage_leaf_max / (1 - r_storage_leaf_max)) * self.DM_structural_leaf * cc_leaf

    @xs.runtime(args=())
    def run_step(self):
        pass

    def finalize_step(self):
        pass

    def finalize(self):
        pass
