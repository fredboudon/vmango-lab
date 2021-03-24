import pathlib
import xsimlab as xs
import numpy as np
import openalea.lpy as lpy

from . import topology


@xs.process
class Geometry:

    lsystem = None

    lstring = xs.foreign(topology.Topology, 'lstring')
    nb_inflo = xs.foreign(topology.Topology, 'nb_inflo')
    seed = xs.foreign(topology.Topology, 'seed')

    phenology = xs.group_dict('phenology')
    growth = xs.group_dict('growth')
    appearance = xs.group_dict('appearance')
    fruit = xs.group_dict('fruit')

    scene = xs.any_object()
    sample_freq = xs.variable(intent='in', static=True, default=-1)
    sampling_steps = np.array([])

    @xs.runtime(args=('nsteps'))
    def initialize(self, nsteps):
        self.rng = np.random.default_rng(seed=self.seed)
        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('geometry.lpy')), {
            'process': self
        })
        self.scene = self.lsystem.sceneInterpretation(self.lstring)
        if self.sample_freq > 0:
            self.sampling_steps = np.linspace(0, nsteps - 1, int(nsteps / self.sample_freq), dtype=np.int32)
        else:
            self.sampling_steps = np.array([0, nsteps - 1])

    @xs.runtime(args=('step', 'nsteps'))
    def run_step(self, step, nsteps):
        if step in self.sampling_steps:
            self.scene = self.lsystem.sceneInterpretation(self.lstring)
