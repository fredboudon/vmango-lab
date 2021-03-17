import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph
import openalea.lpy as lpy
import pathlib

from vmlab.enums import Position, Nature


@xs.process
class Topology:

    archdev = xs.group_dict('arch_dev')
    growth = xs.group_dict('growth')
    rng = xs.global_ref('rng')

    lsystem = None

    GU = xs.index(
        dims='GU',
        global_name='GU'
    )

    lstring = xs.any_object()

    adjacency = xs.variable(
        dims=('GU', 'GU'),
        intent='inout'
    )

    distance = xs.variable(
        dims=('GU', 'GU'),
        intent='out'
    )

    ancestor = xs.variable(
        dims='GU',
        intent='out'
    )
    ancestor_is_apical = xs.variable(
        dims='GU',
        intent='out'
    )
    ancestor_nature = xs.variable(
        dims='GU',
        intent='out'
    )
    position = xs.variable(
        dims='GU',
        intent='out'
    )
    bursted = xs.variable(
        dims='GU',
        intent='out'
    )
    appearance_month = xs.variable(
        dims='GU',
        intent='out'
    )
    appearance_date = xs.variable(
        dims='GU',
        intent='out'
    )
    appeared = xs.variable(
        dims='GU',
        intent='out'
    )
    cycle = xs.variable(
        dims='GU',
        intent='out'
    )

    nb_descendants = xs.variable(
        dims='GU',
        intent='out'
    )

    position = xs.variable(
        dims='GU',
        intent='inout'
    )

    position_parent = xs.variable(
        dims='GU',
        intent='out'
    )

    nb_leaf = xs.variable(
        dims='GU',
        intent='out'
    )

    nb_inflo = xs.variable(
        dims='GU',
        intent='inout'
    )

    nb_fruit = xs.variable(
        dims='GU',
        intent='inout'
    )
    current_cycle = xs.variable(
        intent='inout'
    )
    month_begin_veg_cycle = xs.variable(
        intent='in',
        static=True
    )

    @xs.runtime(args=('nsteps'))
    def initialize(self, nsteps):

        self.GU = np.array([f'GU{x}' for x in range(self.adjacency.shape[0])], dtype=np.dtype('<U10'))
        self.adjacency = np.array(self.adjacency)
        self.distance = csgraph.shortest_path(csgraph.csgraph_from_dense(self.adjacency))
        self.nb_descendants = np.count_nonzero(~np.isinf(self.distance) & (self.distance > 0.), axis=1)
        self.position = np.array(self.position)
        self.position_parent = np.full(self.GU.shape, Position.APICAL)
        self.position_parent[1:] = self.position[np.argwhere(self.adjacency)[:, 0]]
        self.nb_leaf = np.zeros(self.GU.shape)
        self.nb_inflo = np.array(self.nb_inflo)

        self.ancestor = np.array([-1])
        self.ancestor_is_apical = np.array([1.])
        self.ancestor_nature = np.array([Nature.VEGETATIVE])
        self.bursted = np.zeros(self.GU.shape)
        self.appearance_month = np.zeros(self.GU.shape)
        self.appearance_date = np.array(['2002-08-01'], dtype='datetime64[D]')
        self.appeared = np.zeros(self.GU.shape)
        self.cycle = np.full(self.GU.shape, self.current_cycle, dtype=np.float)

        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('topology.lpy')), {
            'process': self,
            'derivation_length': int(nsteps)
        })
        self.lstring = self.lsystem.axiom

    @xs.runtime(args=('step', 'step_start', 'nsteps'))
    def run_step(self, step, step_start, nsteps):

        # set as property of process, needed by lpy
        self.step = step
        self.step_start = step_start
        self.nsteps = nsteps

        self.distance = csgraph.shortest_path(csgraph.csgraph_from_dense(self.adjacency))
        self.nb_descendants = np.count_nonzero(~np.isinf(self.distance) & (self.distance > 0.), axis=1)
        self.nb_leaf = self.growth[('growth', 'nb_internode')]

        self.bursted[:] = 0.
        self.appeared[:] = 0.

        self.bursted[self.archdev[('arch_dev', 'burst_date')] == step_start] = 1.
        day = step_start.astype('datetime64[D]').item()
        self.current_cycle = self.current_cycle + 1 if day.month == self.month_begin_veg_cycle and day.day == 1 else self.current_cycle

        if np.any(self.bursted):
            self.lstring = self.lsystem.derive(self.lstring, step)
