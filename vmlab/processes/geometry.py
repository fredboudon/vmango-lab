import pathlib
import xsimlab as xs
import numpy as np
from pathlib import Path
import io
import openalea.lpy as lpy

from . import (
    topology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class Geometry(ParameterizedProcess):

    lsystem = None
    interpretation_steps = np.array([])

    lstring = xs.foreign(topology.Topology, 'lstring')
    seed = xs.foreign(topology.Topology, 'seed')

    phenology = xs.group_dict('phenology')
    growth = xs.group_dict('growth')
    appearance = xs.group_dict('appearance')
    photosynthesis = xs.group_dict('photosynthesis')

    interpretation_freq = xs.variable(intent='in', static=True, default=-1)
    scene = xs.any_object()
    lpy_parameters = xs.any_object()

    @xs.runtime(args=('nsteps'))
    def initialize(self, nsteps):
        super(Geometry, self).initialize()
        self.rng = np.random.default_rng(seed=self.seed)
        self.lpy_parameters = lpy.lsysparameters.LsystemParameters(
            str(Path(self.parameter_file_path).parent.joinpath(self.parameters.lpy_parameters))
        )
        assert self.lpy_parameters.is_valid()
        with io.open(pathlib.Path(__file__).parent.joinpath('geometry.lpy'), 'r') as file:
            lpy_code = file.read()
            self.lsystem = lpy.Lsystem()
            self.lsystem.set(''.join([lpy_code, self.lpy_parameters.generate_py_code()]), {
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
