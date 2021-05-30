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
    _interpretation_steps = np.nan

    lstring = xs.foreign(topology.Topology, 'lstring')
    seed = xs.foreign(topology.Topology, 'seed')

    phenology = xs.group_dict('phenology')
    harvest = xs.group_dict('harvest')
    growth = xs.group_dict('growth')
    appearance = xs.group_dict('appearance')
    photosynthesis = xs.group_dict('photosynthesis')

    interpretation_freq = xs.variable(
        intent='in',
        static=True,
        default=-1,
        description='If specified as input and "interpretation_steps" in nan \
            a scene will be derived every "interpretation_freq" day'
    )
    interpretation_steps = xs.variable(
        intent='in',
        static=True,
        default=np.nan,
        description='If specified as input \
            a scene exactly "interpretation_steps" scenes will be derived evenly \
            distrubuted over the timespan of the simulation'
    )
    scene = xs.any_object(
        description='A PlantGL Scene instance or None if not derived at teh current step.'
    )
    lpy_parameters = xs.any_object(
        description='An L-Py Parameters instance that holds parameters for processing the lpy file.'
    )

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

        if np.isnan(self.interpretation_steps):
            if self.interpretation_freq > 0:
                self._interpretation_steps = np.linspace(0, nsteps - 1, int(nsteps / self.interpretation_freq), dtype=np.int32)
            elif self.interpretation_freq == -1:
                self._interpretation_steps = np.array([0, nsteps - 1])
            else:
                self._interpretation_steps = np.array([])
        elif type(self.interpretation_steps) == int or type(self.interpretation_steps) == float:
            self._interpretation_steps = np.linspace(0, nsteps - 1, int(self.interpretation_steps), dtype=np.int32)

    @xs.runtime(args=('step', 'nsteps'))
    def run_step(self, step, nsteps):
        if len(self._interpretation_steps) > 0 and (step == 0 or step == nsteps - 1) or step in self._interpretation_steps:
            self.scene = self.lsystem.sceneInterpretation(self.lstring)
        else:
            self.scene = None
