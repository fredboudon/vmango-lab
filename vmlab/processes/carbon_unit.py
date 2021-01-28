import xsimlab as xs
import abc
import numpy as np

from . import fruit_growth
from . import phenology
from . import topology


@xs.process
class CarbonUnit(abc.ABC):
    """Aggregate GUs into CUs
    """

    GU = xs.foreign(topology.Topology, 'GU')
    nb_leaves_gu = xs.foreign(topology.Topology, 'nb_leaves_gu')

    nb_fruits_gu = xs.foreign(fruit_growth.FruitGrowth, 'nb_fruits_gu')
    DM_fruit_max_fruit_gu = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_max_gu')
    DM_fruit_0_fruit_gu = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_0_gu')

    dd_delta_gu = xs.foreign(phenology.Phenology, 'dd_delta_gu')
    dd_cum_gu = xs.foreign(phenology.Phenology, 'dd_cum_gu')

    CU = xs.index(('CU'), global_name='CU')

    nb_leaves = xs.variable(
        dims=('CU'),
        intent='out',
        global_name='nb_leaves'
    )

    nb_fruits = xs.variable(
        dims=('CU'),
        intent='out',
        global_name='nb_fruits'
    )

    DM_fruit_max = xs.variable(
        dims=('CU'),
        intent='out',
        description='potential total maximal fruit dry mass per CU',
        attrs={
            'unit': 'g DM'
        },
        global_name='DM_fruit_max'
    )

    DM_fruit_0 = xs.variable(
        dims=('CU'),
        intent='out',
        description='fruit dry mass per CU at the end of cell division (at 352.72 dd)',
        attrs={
            'unit': 'g DM'
        },
        static=True,
        global_name='DM_fruit_0'
    )

    dd_delta = xs.variable(
        dims=('CU'),
        intent='out',
        description='daily variation in degree days (avg per CU)',
        attrs={
            'unit': 'dd day-1'
        },
        global_name='dd_delta'
    )

    dd_cum = xs.variable(
        dims=('CU'),
        intent='out',
        description='cumulated degree-days of the current day after bloom date (avg per CU)',
        attrs={
            'unit': 'dd'
        },
        global_name='dd_cum'
    )

    CUxGU = xs.variable(
        dims=('CU', 'GU'),
        intent='out',
        global_name='CUxGU',
    )

    # @CUxGU.validator
    # def _validator(prc, var, val):
    #     print(np.sum(val), len(val))
    #     if np.sum(val) != len(val):
    #         raise ValueError()

    @abc.abstractmethod
    def update_index(self, step):
        pass

    @abc.abstractmethod
    def update_mapping(self, step):
        pass

    def initialize(self):

        self.update_index(-1)

        # CUs in rows, GUs in columns, therefor all ops over columns dim=1
        self.update_mapping(-1)
        self.nb_leaves = np.sum(self.CUxGU * self.nb_leaves_gu, 1)
        self.nb_fruits = np.sum(self.CUxGU * self.nb_fruits_gu, 1)
        self.DM_fruit_max = np.sum(self.CUxGU * self.DM_fruit_max_fruit_gu * self.nb_fruits_gu, 1)
        self.DM_fruit_0 = np.sum(self.CUxGU * self.DM_fruit_0_fruit_gu * self.nb_fruits_gu, 1)
        self.dd_delta = np.mean(self.CUxGU * self.dd_delta_gu, 1)
        self.dd_cum = np.mean(self.CUxGU * self.dd_cum_gu, 1)

    @xs.runtime(args=('nsteps', 'step', 'step_start', 'step_end', 'step_delta'))
    def run_step(self, nsteps, step, step_start, step_end, step_delta):
        self.nb_leaves = np.sum(self.CUxGU * self.nb_leaves_gu, 1)
        self.nb_fruits = np.sum(self.CUxGU * self.nb_fruits_gu, 1)
        self.DM_fruit_max = np.sum(self.CUxGU * self.DM_fruit_max_fruit_gu * self.nb_fruits_gu, 1)

        DM_fruit_0 = self.CUxGU * self.DM_fruit_0_fruit_gu * self.nb_fruits_gu
        DM_fruit_0[DM_fruit_0 == 0] = np.nan
        self.DM_fruit_0 = np.nan_to_num(np.nanmean(DM_fruit_0, 1), copy=False)

        dd_delta = self.CUxGU * self.dd_delta_gu * (self.nb_fruits_gu > 0)
        dd_delta[dd_delta == 0] = np.nan
        self.dd_delta = np.nan_to_num(np.nanmean(dd_delta, 1), copy=False)

        dd_cum = self.CUxGU * self.dd_cum_gu * (self.nb_fruits_gu > 0)
        dd_cum[dd_cum == 0] = np.nan
        self.dd_cum = np.nan_to_num(np.nanmean(dd_cum, 1), copy=False)


@xs.process
class Identity(CarbonUnit):
    """Map GU 1:1 to a CU
    """

    def update_index(self, step):
        if step < 0:
            self.CU = np.array([f'CU{x}' for x in range(len(self.GU))], dtype=np.dtype('<U10'))

    def update_mapping(self, step):
        if step < 0:
            self.CUxGU = np.identity(self.GU.shape[0])


@xs.process
class JustOne(CarbonUnit):
    """Merge all GU into one CU
    """

    def update_index(self, step):
        if step < 0:
            self.CU = np.array(['CU_One_And_Only'], dtype=np.dtype('<U20'))

    def update_mapping(self, step):
        if step < 0:
            self.CUxGU = np.ones((1, self.GU.shape[0]))
