import xsimlab as xs
import numpy as np
import datetime

from . import parameters
from . import growth_unit_growth


@xs.process
class InfloGrowth():

    params = xs.foreign(parameters.Parameters, 'inflo_growth')

    GU = xs.foreign(growth_unit_growth.GrowthUnitGrowth, 'GU')

    bloom_date = xs.variable(
        dims=('GU'),
        intent='inout',
        description='bloom date',
        attrs={
            'unit': 'date'
        },
        static=True
    )

    DAB = xs.variable(
        dims=('GU'),
        intent='out',
        description='days after bloom',
        attrs={
            'unit': 'd'
        }
    )

    def initialize(self):
        self.bloom_date = np.array([np.datetime64(datetime.date.fromisoformat(bloom_date)).astype('datetime64[D]')
                                    for bloom_date in self.bloom_date])
        self.DAB = np.ones(self.GU.shape) * -1

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        self.DAB = np.where(
            step_start >= self.bloom_date,
            (step_start - self.bloom_date).astype('timedelta64[D]') / np.timedelta64(1, 'D'),
            -1
        )

    def finalize_step(self):
        pass

    def finalize(self):
        pass
