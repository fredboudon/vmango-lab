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

    ripeness_index = xs.variable(
        dims=('GU'),
        intent='inout',
        description='ripening index',
        attrs={
            'unit': '-'
        },
        global_name='ripeness_index'
    )

    def initialize(self):
        super(Harvest, self).initialize()
        self.ripeness_index = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):
        Tthresh_fruit_ripe = self.parameters.Tthresh_fruit_ripe
        self.ripeness_index[self.fruit_growth_tts > 0.] = np.minimum(1.0, self.fruit_growth_tts[self.fruit_growth_tts > 0.] / Tthresh_fruit_ripe)
