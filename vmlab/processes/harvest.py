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
        description='Just a helper to set those GUs to 1. where fruits are ripe and harvested that day',
        groups='harvest',
        attrs={
            'unit': '-'
        }
    )

    nb_fruit_harvested = xs.variable(
        dims=('GU'),
        intent='inout',
        description='Number of fruits harvested',
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
        growing = self.fruit_growth_tts > 0.
        Tthresh_fruit_ripe = self.parameters.Tthresh_fruit_ripe
        self.ripeness_index[growing] = np.minimum(1.0, self.fruit_growth_tts[growing] / Tthresh_fruit_ripe)
        self.harvested[(self.nb_fruit_harvested == 0) & (self.ripeness_index == 1.)] = 1.
        self.nb_fruit_harvested[self.harvested == 1.] = self.nb_fruit[self.harvested == 1.]


@xs.process
class HarvestByQuality(Harvest):
    """Decide when fruit is ripe based of fruit quality
    """

    fruit_quality = xs.group_dict('fruit_quality')

    def initialize(self):
        super(HarvestByQuality, self).initialize()

    @xs.runtime(args=())
    def run_step(self):
        sucrose = self.fruit_quality[('fruit_quality', 'sucrose')]
        self.harvested[:] = 0.
        growing = self.fruit_growth_tts > 0.
        sucrose_thresh_fruit_ripe = self.parameters.sucrose_thresh_fruit_ripe
        self.ripeness_index[growing] = np.where(
            self.ripeness_index[growing] < (sucrose[growing] / sucrose_thresh_fruit_ripe),
            np.minimum(1., sucrose[growing] / sucrose_thresh_fruit_ripe),
            self.ripeness_index[growing]
        )
        self.harvested[(self.nb_fruit_harvested == 0) & (self.ripeness_index == 1.)] = 1.
        self.nb_fruit_harvested[self.harvested == 1.] = self.nb_fruit[self.harvested == 1.]
