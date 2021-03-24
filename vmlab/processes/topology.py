import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph
import openalea.lpy as lpy
import pathlib


@xs.process
class Topology:

    archdev = xs.group_dict('arch_dev')
    # numpy random generator
    rng = xs.any_object(global_name='rng')
    seed = xs.variable(default=0)

    lsystem = None

    GU = xs.index(dims='GU', global_name='GU')

    lstring = xs.any_object()

    current_cycle = xs.variable(intent='inout')
    month_begin_veg_cycle = xs.variable(intent='in', static=True)
    doy_begin_flowering = xs.variable(intent='in', static=True)

    adjacency = xs.variable(dims=('GU', 'GU'), intent='inout')
    ancestor_is_apical = xs.variable(dims='GU', intent='inout')
    ancestor_nature = xs.variable(dims='GU', intent='inout')
    is_apical = xs.variable(dims='GU', intent='inout')
    appearance_month = xs.variable(dims='GU', intent='inout')
    appearance_date = xs.variable(dims='GU', intent='inout')
    cycle = xs.variable(dims='GU', intent='inout')

    distance = xs.variable(dims=('GU', 'GU'), intent='out')
    bursted = xs.variable(dims='GU', intent='out')
    appeared = xs.variable(dims='GU', intent='out')
    flowered = xs.variable(dims='GU', intent='out')
    nb_descendants = xs.variable(dims='GU', intent='out')
    ancestor = xs.variable(dims='GU', intent='out')
    parent_is_apical = xs.variable(dims='GU', intent='out')
    nb_inflo = xs.variable(dims='GU', intent='out')
    nb_fruit = xs.variable(dims='GU', intent='out')

    @xs.runtime(args=('nsteps', 'step_start'))
    def initialize(self, nsteps, step_start):

        self.rng = np.random.default_rng(self.seed)

        self.adjacency = np.array(self.adjacency, dtype=np.float32)
        self.GU = np.array([f'GU{x}' for x in range(self.adjacency.shape[0])], dtype=np.dtype('<U10'))

        self.ancestor_is_apical = np.array(self.ancestor_is_apical, dtype=np.float32)
        self.ancestor_nature = np.array(self.ancestor_nature, dtype=np.float32)
        self.is_apical = np.array(self.is_apical, dtype=np.float32)
        self.appearance_month = np.array(self.appearance_month, dtype=np.float32)
        self.cycle = np.array(self.cycle, dtype=np.float32)
        self.appearance_date = np.array(self.appearance_date, dtype='datetime64[D]')

        self.distance = csgraph.shortest_path(csgraph.csgraph_from_dense(self.adjacency))
        self.nb_descendants = np.count_nonzero(~np.isinf(self.distance) & (self.distance > 0.), axis=1)
        self.parent_is_apical = np.full(self.GU.shape, 1.)
        self.ancestor = np.full(self.GU.shape, 0.)
        self.parent_is_apical[1:] = self.is_apical[np.argwhere(self.adjacency)[:, 0]]

        self.nb_leaf = np.zeros(self.GU.shape, dtype=np.float32)
        self.nb_inflo = np.zeros(self.GU.shape, dtype=np.float32)
        self.nb_fruit = np.zeros(self.GU.shape, dtype=np.float32)

        self.bursted = np.zeros(self.GU.shape, dtype=np.float32)
        self.appeared = np.zeros(self.GU.shape, dtype=np.float32)
        self.flowered = np.zeros(self.GU.shape, dtype=np.float32)

        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('topology.lpy')), {
            'process': self,
            'derivation_length': int(nsteps)
        })
        self.lstring = self.lsystem.derive(self.lsystem.axiom, 0, 1)

    @xs.runtime(args=('step', 'step_start', 'nsteps'))
    def run_step(self, step, step_start, nsteps):

        # set as property of process, needed by lpy
        self.step = step
        self.step_start = step_start
        self.nsteps = nsteps

        self.bursted[:] = 0.
        if step == 0:
            self.appeared = ((self.cycle == np.max(self.cycle)) & (self.nb_descendants == 0.)).astype(np.float)
        else:
            self.appeared[:] = 0.
        self.flowered[:] = 0.

        # make boolean. these are just helpers and do not need to appear in sim. output
        self.bursted[self.archdev[('arch_dev', 'pot_burst_date')] == step_start] = 1.
        self.flowered[self.archdev[('arch_dev', 'pot_flowering_date')] == step_start] = 1.
        self.nb_inflo[self.flowered == 1.] = self.archdev[('arch_dev', 'pot_nb_inflo')][self.flowered == 1.]
        self.nb_fruit[self.flowered == 1.] = self.archdev[('arch_dev', 'pot_nb_fruit')][self.flowered == 1.]

        day = step_start.astype('datetime64[D]').item()
        self.current_cycle = self.current_cycle + 1 if day.month == self.month_begin_veg_cycle and day.day == 1 else self.current_cycle

        if np.any(self.bursted):
            total_nb_children = np.sum(self.archdev[('arch_dev', 'pot_nb_lateral_children')][self.bursted == 1.] + self.archdev[('arch_dev', 'pot_has_apical_child')][self.bursted == 1.])
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
