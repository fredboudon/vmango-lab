import pathlib
import xsimlab as xs
import numpy as np
import openalea.lpy as lpy
import openalea.plantgl.all as pgl

from . import topology


@xs.process
class Geometry:

    lsystem = None

    lstring = xs.foreign(topology.Topology, 'lstring')
    phenology = xs.group_dict('phenology')
    growth = xs.group_dict('growth')

    scene = xs.any_object()

    def initialize(self):

        self.scene = pgl.Scene()

    @xs.runtime(args=())
    def run_step(self):

        if not self.lsystem:
            self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('geometry.lpy')), {
                'process': self
            })

        self.scene = self.lsystem.sceneInterpretation(self.lstring)
