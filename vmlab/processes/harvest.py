import xsimlab as xs
import numpy as np

from . import (
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class Harvest(ParameterizedProcess):
    """Decide when fruit is ripe
    """

    nb_gu = xs.global_ref('nb_gu')
    fruit_growth_tts = xs.foreign(phenology.Phenology, 'fruit_growth_tts')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')

    ripeness_index = xs.variable(
        dims=('GU'),
        intent='inout',
        description='ripening index',
        groups='harvest',
        attrs={
            'unit': '-'
        }
    )

    harvested = xs.variable(
        dims=('GU'),
        intent='inout',
        description='',
        groups='harvest',
        attrs={
            'unit': '-'
        }
    )

    nb_fruit_harvested = xs.variable(
        dims=('GU'),
        intent='inout',
        description='ripening index',
        groups='harvest',
        attrs={
            'unit': '-'
        }
    )

    def initialize(self):
        super(Harvest, self).initialize()
        self.ripeness_index = np.zeros(self.nb_gu, dtype=np.float32)
        self.harvested = np.zeros(self.nb_gu, dtype=np.float32)
        self.nb_fruit_harvested = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):
        self.harvested[:] = 0.
        Tthresh_fruit_ripe = self.parameters.Tthresh_fruit_ripe
        self.ripeness_index[self.fruit_growth_tts > 0.] = np.minimum(1.0, self.fruit_growth_tts[self.fruit_growth_tts > 0.] / Tthresh_fruit_ripe)
        self.harvested[(self.nb_fruit_harvested == 0) & (self.ripeness_index == 1.)] = 1.
        self.nb_fruit_harvested[self.harvested == 1.] = self.nb_fruit[self.harvested == 1.]
