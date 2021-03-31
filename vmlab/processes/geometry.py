import pathlib
import xsimlab as xs
import numpy as np
import openalea.lpy as lpy

from . import (
    topology
)


@xs.process
class Geometry:

    lsystem = None

    lstring = xs.foreign(topology.Topology, 'lstring')
    seed = xs.foreign(topology.Topology, 'seed')

    phenology = xs.group_dict('phenology')
    growth = xs.group_dict('growth')
    appearance = xs.group_dict('appearance')

    scene = xs.any_object()
    interpretation_freq = xs.variable(intent='in', static=True, default=-1)
    interpretation_steps = np.array([])

    @xs.runtime(args=('nsteps'))
    def initialize(self, nsteps):
        self.rng = np.random.default_rng(seed=self.seed)
        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('geometry.lpy')), {
            'process': self
        })
        self.scene = self.lsystem.sceneInterpretation(self.lstring)
        if self.interpretation_freq > 0:
            self.interpretation_steps = np.linspace(0, nsteps - 1, int(nsteps / self.interpretation_freq), dtype=np.int32)
        elif self.interpretation_freq == -1:
            self.interpretation_steps = np.array([0, nsteps - 1])

    @xs.runtime(args=('step', 'nsteps'))
    def run_step(self, step, nsteps):
        if (self.growth[('growth', 'any_is_growing')] or self.interpretation_freq == -1) and step in self.interpretation_steps:
            self.scene = self.lsystem.sceneInterpretation(self.lstring)
