import xsimlab as xs
import abc
import numpy as np

from . import growth_unit_growth
from . import fruit_growth


@xs.process
class Branching(abc.ABC):
    """Aggregate GUs into fruiting branches
    """

    GU = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'GU')
    nb_leaves_gu = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'nb_leaves')

    nb_fruits_gu = xs.foreign(fruit_growth.FruitGrowth, 'nb_fruits')
    DM_fruit_max_fruit = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_max')
    DM_fruit_0_fruit = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_0')
    dd_delta_gu = xs.foreign(fruit_growth.FruitGrowth, 'dd_delta')
    dd_cum_gu = xs.foreign(fruit_growth.FruitGrowth, 'dd_cum')

    branch = xs.index(('branch'))

    nb_leaves = xs.variable(
        dims=('branch'),
        intent='out'
    )

    nb_fruits = xs.variable(
        dims=('branch'),
        intent='out'
    )

    DM_fruit_max = xs.variable(
        dims=('branch'),
        intent='out',
        description='potential total maximal fruit dry mass per branch',
        attrs={
            'unit': 'g DM'
        }
    )

    DM_fruit_0 = xs.variable(
        dims=('branch'),
        intent='out',
        description='fruit dry mass per branch at the end of cell division (at 352.72 dd)',
        attrs={
            'unit': 'g DM'
        },
        static=True
    )

    dd_delta = xs.variable(
        dims=('branch'),
        intent='out',
        description='daily variation in degree days (avg per branch)',
        attrs={
            'unit': 'dd day-1'
        }
    )

    dd_cum = xs.variable(
        dims=('branch'),
        intent='out',
        description='cumulated degree-days of the current day after bloom date (avg per branch)',
        attrs={
            'unit': 'dd'
        }
    )

    branches = xs.variable(
        dims=('branch', 'GU'),
        intent='out'
    )

    @branches.validator
    def _validator(prc, var, val):
        if np.sum(val) != len(val):
            raise ValueError()

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def run_step(self):
        pass


@xs.process
class Identity(Branching):
    """Map GU 1:1 to a branch
    """

    def initialize(self):

        self.branch = np.arange(self.GU.shape[0])

        # branches in rows, GUs in columns, therefor all ops over columns dim=1
        self.branches = np.identity(self.GU.shape[0])
        self.nb_leaves = np.sum(self.branches * self.nb_leaves_gu, 1)
        self.nb_fruits = np.sum(self.branches * self.nb_fruits_gu, 1)
        self.DM_fruit_max = np.sum(self.branches * self.DM_fruit_max_fruit * self.nb_fruits_gu, 1)
        self.DM_fruit_0 = np.sum(self.branches * self.DM_fruit_0_fruit * self.nb_fruits_gu, 1)
        self.dd_delta = np.mean(self.branches * self.dd_delta_gu, 1)
        self.dd_cum = np.mean(self.branches * self.dd_cum_gu, 1)

    @xs.runtime(args=('step'))
    def run_step(self, step):

        self.nb_leaves = np.sum(self.branches * self.nb_leaves_gu, 1)
        self.nb_fruits = np.sum(self.branches * self.nb_fruits_gu, 1)
        self.DM_fruit_max = np.sum(self.branches * self.DM_fruit_max_fruit * self.nb_fruits_gu, 1)

        DM_fruit_0 = self.branches * self.DM_fruit_0_fruit * self.nb_fruits_gu
        DM_fruit_0[DM_fruit_0 == 0] = np.nan
        self.DM_fruit_0 = np.nan_to_num(np.nanmean(DM_fruit_0, 1), copy=False)

        dd_delta = self.branches * self.dd_delta_gu
        dd_delta[dd_delta == 0] = np.nan
        self.dd_delta = np.nan_to_num(np.nanmean(dd_delta, 1), copy=False)

        dd_cum = self.branches * self.dd_cum_gu
        dd_cum[dd_cum == 0] = np.nan
        self.dd_cum = np.nan_to_num(np.nanmean(dd_cum, 1), copy=False)
