import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph
import openalea.lpy as lpy
import pathlib

from vmlab.enums import Position, Nature


@xs.process
class Topology:

    archdev = xs.group_dict('arch_dev')
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
    flowered = xs.variable(
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

    nb_inflo = xs.variable(
        dims='GU',
        intent='out'
    )

    nb_fruit = xs.variable(
        dims='GU',
        intent='out'
    )
    current_cycle = xs.variable(
        intent='inout'
    )
    month_begin_veg_cycle = xs.variable(
        intent='in',
        static=True
    )
    doy_begin_flowering = xs.variable(intent='in', static=True)

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
        self.nb_inflo = np.zeros(self.GU.shape)
        self.nb_fruit = np.zeros(self.GU.shape)

        self.ancestor = np.array([-1])
        self.ancestor_is_apical = np.array([1.])
        self.ancestor_nature = np.array([Nature.VEGETATIVE])
        self.bursted = np.zeros(self.GU.shape)
        self.appearance_month = np.zeros(self.GU.shape)
        self.appearance_date = np.array(['2002-08-01'], dtype='datetime64[D]')
        self.appeared = np.zeros(self.GU.shape)
        self.flowered = np.zeros(self.GU.shape)
        self.cycle = np.full(self.GU.shape, self.current_cycle, dtype=np.float)

        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('topology.lpy')), {
            'process': self,
            'derivation_length': int(nsteps)
        })
        self.lstring = self.lsystem.derive(self.lsystem.axiom, 1, 1)

    @xs.runtime(args=('step', 'step_start', 'nsteps'))
    def run_step(self, step, step_start, nsteps):

        # set as property of process, needed by lpy
        self.step = step
        self.step_start = step_start
        self.nsteps = nsteps

        self.bursted[:] = 0.
        self.appeared[:] = 0.
        self.flowered[:] = 0.

        # make boolean. these are just helpers and do not need to appear in sim. output
        self.bursted[self.archdev[('arch_dev', 'burst_date')] == step_start] = 1.
        self.flowered[self.archdev[('flowering_week', 'flowering_date')] == step_start] = 1.
        self.nb_inflo[self.flowered == 1.] = self.archdev[('nb_inflorescences', 'nb_inflorescences')][self.flowered == 1.]

        day = step_start.astype('datetime64[D]').item()
        self.current_cycle = self.current_cycle + 1 if day.month == self.month_begin_veg_cycle and day.day == 1 else self.current_cycle

        if np.any(self.bursted):
            total_nb_children = np.sum(self.archdev[('arch_dev', 'nb_lateral_children')][self.bursted == 1.] + self.archdev[('arch_dev', 'has_apical_child')][self.bursted == 1.])
            self.idx_first_child = self.GU.shape[0]
            self.GU = np.append(self.GU, [f'GU{i + self.GU.shape[0]}' for i in range(int(total_nb_children))])
            # initialize new GUs
            step_date = step_start.astype('datetime64[D]')
            self.appearance_month[self.idx_first_child:] = step_date.item().month
            self.appearance_date[self.idx_first_child:] = step_date
            self.appeared[self.idx_first_child:] = 1.
            self.adjacency[np.isnan(self.adjacency)] = 0.
            self.cycle[self.idx_first_child:] = self.current_cycle
            self.distance = csgraph.shortest_path(csgraph.csgraph_from_dense(self.adjacency))
            self.nb_descendants = np.count_nonzero(~np.isinf(self.distance) & (self.distance > 0.), axis=1)
            self.lstring = self.lsystem.derive(self.lstring, step, 1)
