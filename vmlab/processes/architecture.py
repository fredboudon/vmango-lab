import xsimlab as xs
import numpy as np

from . import parameters


@xs.process
class Architecture():

    params = xs.foreign(parameters.Parameters, 'architecture')

    # LFratio = xs.variable(
    #     dims=('GU'),
    #     intent='out'
    # )

    # LA = xs.variable(
    #     dims=('GU'),
    #     intent='out',
    #     description='total leaf area per branch',
    #     attrs={
    #         'unit': 'm²'
    #     }
    # )

    def initialize(self):

        self.GU = np.arange(0, len(self.nb_fruits), step=1, dtype=np.int64)
        self.nb_fruits_ini = np.array(self.nb_fruits, dtype=np.int64)
        self.nb_fruits = np.zeros(self.nb_fruits_ini.shape, dtype=np.int64)
        # self.LFratio = self.nb_leaves / self.nb_fruits

    @xs.runtime(args=('step'))
    def run_step(self, step):
        pass
        # leaf area (eq. 11) :
        # self.LA = self.params[1].e_nleaf2LA_1 * self.LFratio ** self.params[1].e_nleaf2LA_2
