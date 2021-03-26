import xsimlab as xs
import numpy as np

from ._base.parameter import ParameterizedProcess


@xs.process
class HarvestProcess(ParameterizedProcess):
    """Decide when fruit is ripe
    """

    ripeness_index = xs.variable(
        dims=('GU'),
        intent='out',
        description='ripening index',
        attrs={
            'unit': '-'
        }
    )

    def initialize(self):
        super(HarvestProcess, self).initialize()

    @xs.runtime(args=())
    def run_step(self):
        sucrose_ripe_thresh = self.parameters.sucrose_ripe_thresh
        self.ripe = np.minimum(1.0, self.sucrose / sucrose_ripe_thresh)

    def finalize_step(self):
        pass

    def finalize(self):
        pass
