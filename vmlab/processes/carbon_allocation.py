import xsimlab as xs
import numpy as np

from . import fruit_growth
from . import phenology
from . import topology


@xs.process
class CarbonAllocation:
    """Aggregate GUs into CUs
    """

    GU = xs.foreign(topology.Topology, 'GU')
    nb_leaves_gu = xs.foreign(topology.Topology, 'nb_leaves_gu')

    nb_fruits_gu = xs.foreign(fruit_growth.FruitGrowth, 'nb_fruits_gu')
    DM_fruit_max_fruit_gu = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_max_gu')
    DM_fruit_0_fruit_gu = xs.foreign(fruit_growth.FruitGrowth, 'DM_fruit_0_gu')

    dd_delta_gu = xs.foreign(phenology.FlowerPhenology, 'dd_delta_gu')
    dd_cum_gu = xs.foreign(phenology.FlowerPhenology, 'dd_cum_gu')

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

    def initialize(self):

        self.CU = np.array([f'CU{x}' for x in range(len(self.GU))], dtype=np.dtype('<U10'))
        self.CUxGU = np.identity(self.GU.shape[0])

        # CUs in rows, GUs in columns, therefor all ops over columns dim=1
        self.nb_leaves = np.sum(self.CUxGU * self.nb_leaves_gu, 1)
        self.nb_fruits = np.sum(self.CUxGU * self.nb_fruits_gu, 1)
        self.DM_fruit_max = np.sum(self.CUxGU * self.DM_fruit_max_fruit_gu * self.nb_fruits_gu, 1)
        self.DM_fruit_0 = np.sum(self.CUxGU * self.DM_fruit_0_fruit_gu * self.nb_fruits_gu, 1)
        self.dd_delta = np.mean(self.CUxGU * self.dd_delta_gu, 1)
        self.dd_cum = np.mean(self.CUxGU * self.dd_cum_gu, 1)

    @xs.runtime(args=())
    def run_step(self):

        self.nb_leaves = np.sum(self.CUxGU * self.nb_leaves_gu, 1)
        self.nb_fruits = np.sum(self.CUxGU * self.nb_fruits_gu, 1)
        self.DM_fruit_max = np.sum(self.CUxGU * self.DM_fruit_max_fruit_gu * self.nb_fruits_gu, 1)

        DM_fruit_0 = self.CUxGU * self.DM_fruit_0_fruit_gu * self.nb_fruits_gu
        DM_fruit_0[DM_fruit_0 == 0] = np.nan
        if not np.all(np.isnan(DM_fruit_0)):
            self.DM_fruit_0 = np.nan_to_num(np.nanmean(DM_fruit_0, 1), copy=False)

        dd_delta = self.CUxGU * self.dd_delta_gu * (self.nb_fruits_gu > 0)
        dd_delta[dd_delta == 0] = np.nan
        if not np.all(np.isnan(dd_delta)):
            self.dd_delta = np.nan_to_num(np.nanmean(dd_delta, 1), copy=False)

        dd_cum = self.CUxGU * self.dd_cum_gu * (self.nb_fruits_gu > 0)
        dd_cum[dd_cum == 0] = np.nan
        if not np.all(np.isnan(dd_cum)):
            self.dd_cum = np.nan_to_num(np.nanmean(dd_cum, 1), copy=False)
