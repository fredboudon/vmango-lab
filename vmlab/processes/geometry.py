import pathlib
import xsimlab as xs
import openalea.lpy as lpy

from . import topology


@xs.process
class GeometryVegetative:

    lsystem = None

    lstring = xs.foreign(topology.Topology, 'lstring')

    phenology = xs.group_dict('phenology')
    growth = xs.group_dict('growth')
    appearance = xs.group_dict('appearance')

    scene = xs.any_object()
    sampling_rate = xs.variable(intent='in', static=True)

    @xs.runtime(args=())
    def initialize(self):
        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('geometry_vegetative.lpy')), {
            'process': self
        })

    @xs.runtime(args=('step'))
    def run_step(self, step):
        if step % self.sampling_rate == 0:
            self.scene = self.lsystem.sceneInterpretation(self.lstring)


@xs.process
class Geometry:

    lsystem = None

    lstring = xs.foreign(topology.Topology, 'lstring')
    nb_inflo = xs.foreign(topology.Topology, 'nb_inflo')

    phenology = xs.group_dict('phenology')
    growth = xs.group_dict('growth')
    appearance = xs.group_dict('appearance')
    fruit = xs.group_dict('fruit')

    scene = xs.any_object()
    sampling_rate = xs.variable(intent='in', static=True)

    @xs.runtime(args=())
    def initialize(self):
        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('geometry.lpy')), {
            'process': self
        })

    @xs.runtime(args=('step'))
    def run_step(self, step):
        if step % self.sampling_rate == 0:
            self.scene = self.lsystem.sceneInterpretation(self.lstring)
