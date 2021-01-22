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

    branch = xs.index(('branch'))

    nb_leaves = xs.variable(
        dims=('branch'),
        intent='out'
    )

    nb_fruits = xs.variable(
        dims=('branch'),
        intent='out'
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

        self.branch = np.arange(0, len(self.GU), step=1)
        self.branches = np.identity(len(self.GU))

    @xs.runtime(args=('step'))
    def run_step(self, step):
        pass
