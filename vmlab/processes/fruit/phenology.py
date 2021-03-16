import xsimlab as xs
import numpy as np

from . import environment
from . import topology
from ._base.parameter import BaseParameterizedProcess


@xs.process
class LeafPhenology(BaseParameterizedProcess):

    GU = xs.foreign(topology.Topology, 'GU')

    TM = xs.foreign(environment.Environment, 'TM')

    leaf_growth_tts = xs.variable(
        dims=('GU'),
        intent='out'
    )

    def initialize(self):

        super(LeafPhenology, self).initialize()

        self.leaf_growth_tts = np.zeros(self.GU.shape)

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters
        Tbase_leaf = params.Tbase_leaf

        self.leaf_growth_tts += max(0, self.TM - Tbase_leaf)

    def finalize_step(self):
        pass

    def finalize(self):
        pass


@xs.process
class GrowthUnitPhenology(BaseParameterizedProcess):

    stage_tbase = np.array([100, 200, 300, np.inf])
    stage_name = np.array(['A', 'B', 'C', 'D'])

    GU = xs.foreign(topology.Topology, 'GU')

    TM = xs.foreign(environment.Environment, 'TM')

    bursts = xs.group_dict('bursts')

    gu_growth_tts = xs.variable(
        dims=('GU'),
        intent='out'
    )

    gu_stage = xs.variable(
        dims=('GU'),
        intent='out',
        default='A'
    )

    def initialize(self):

        super(GrowthUnitPhenology, self).initialize()

        self.gu_growth_tts = np.zeros(self.GU.shape)
        self.gu_stage = np.array(['A' for _ in range(self.GU.shape[0])])

    @xs.runtime(args=())
    def run_step(self):

        params = self.parameters
        Tbase_gu = params.Tbase_gu

        self.gu_stage[self.gu_stage == ''] = self.stage_name[0]

        self.gu_growth_tts += max(0, self.TM - Tbase_gu)

        masks = []
        for stage, t in zip(self.stage_name, self.stage_tbase):
            masks.append((self.gu_stage == stage) & (self.gu_growth_tts >= t))

        for i, mask in enumerate(masks):
            self.gu_stage[mask] = self.stage_name[i+1] if i < len(self.stage_name) - 1 else self.stage_name[-1]
            self.gu_growth_tts[mask] = 0.

        # append previously bursted because burst process comes after phenology
        nb_bursted = np.count_nonzero(self.bursts[('gu_burst', 'gu_bursted')])
        if nb_bursted > 0:
            gu_bursted = np.append(self.bursts[('gu_burst', 'gu_bursted')], np.zeros(nb_bursted, dtype=np.bool))
            self.gu_growth_tts[gu_bursted] = 0.

    def finalize_step(self):
        pass

    def finalize(self):
        pass


@xs.process
class FlowerPhenology(BaseParameterizedProcess):

    GU = xs.foreign(topology.Topology, 'GU')

    TM = xs.foreign(environment.Environment, 'TM')

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
            'unit': 'day'
        }
    )

    dd_cum = xs.variable(
        dims=('GU'),
        intent='out',
        description='cumulated degree-days of the current day after bloom date',
        attrs={
            'unit': 'dd'
        }
    )

    dd_delta = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily variation in degree days',
        attrs={
            'unit': 'dd day-1'
        }
    )

    def initialize(self):

        super(FlowerPhenology, self).initialize()

        self.bloom_date = np.array(self.bloom_date, dtype='datetime64[D]')
        self.DAB = np.zeros(self.GU.shape)
        self.dd_delta = np.zeros(self.GU.shape)
        self.dd_cum = np.zeros(self.GU.shape)

    @xs.runtime(args=('step_start'))
    def run_step(self, step_start):

        params = self.parameters
        Tbase_fruit_growth = params.Tbase_fruit_growth

        self.DAB = np.where(
            step_start >= self.bloom_date,
            (step_start - self.bloom_date).astype('timedelta64[D]') / np.timedelta64(1, 'D'),
            -1
        )

        self.dd_delta = np.where(
            step_start >= self.bloom_date,
            max(0, self.TM - Tbase_fruit_growth),
            0.
        )

        self.dd_cum = np.where(
            step_start >= self.bloom_date,
            self.dd_cum + self.dd_delta,
            0.
        )

    def finalize_step(self):
        pass

    def finalize(self):
        pass
