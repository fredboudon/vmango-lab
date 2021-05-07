import xsimlab as xs
import numpy as np
from scipy.sparse import csgraph
import openalea.lpy as lpy
import pathlib

from ._base.parameter import ParameterizedProcess


@xs.process
class Topology(ParameterizedProcess):

    lsystem = None

    archdev = xs.group_dict('arch_dev')

    seed = xs.variable(default=0, static=True, global_name='seed')
    month_begin_veg_cycle = xs.variable(default=7, intent='in', static=True)
    doy_begin_flowering = xs.variable(default=214, intent='in', static=True)

    sim_start_date = xs.variable(intent='inout', static=True)
    GU = xs.index(dims='GU')
    GU_ = xs.index(dims='GU_')
    # make an inout variable so it is ignored in process ordering
    nb_gu = xs.variable(intent='inout', default=0, static=True, global_name='nb_gu')

    lstring = xs.any_object()

    current_cycle = xs.variable(intent='inout')
    adjacency = xs.variable(dims=('GU', 'GU_'), intent='inout')
    ancestor_is_apical = xs.variable(dims='GU', intent='inout')
    ancestor_nature = xs.variable(dims='GU', intent='inout')
    is_apical = xs.variable(dims='GU', intent='inout')
    appearance_month = xs.variable(dims='GU', intent='inout')
    cycle = xs.variable(dims='GU', intent='inout')

    appearance_date = xs.variable(dims='GU', intent='out')
    distance = xs.variable(dims=('GU', 'GU_'), intent='out')
    bursted = xs.variable(dims='GU', intent='out')
    appeared = xs.variable(dims='GU', intent='out')
    nb_descendants = xs.variable(dims='GU', intent='out')
    ancestor = xs.variable(dims='GU', intent='out')
    parent_is_apical = xs.variable(dims='GU', intent='out')
    is_initially_terminal = xs.variable(dims='GU', intent='out')

    @xs.runtime(args=('nsteps', 'step_start'))
    def initialize(self, nsteps, step_start):

        super(Topology, self).initialize()

        self.sim_start_date = np.datetime64(self.sim_start_date)
        self.adjacency = np.array(self.adjacency, dtype=np.float32)
        self.GU = np.array([x for x in range(self.adjacency.shape[0])], dtype=np.int32)
        self.GU_ = self.GU
        self.nb_gu = self.GU.shape[0]

        self.ancestor_is_apical = np.array(self.ancestor_is_apical, dtype=np.float32)
        self.ancestor_nature = np.array(self.ancestor_nature, dtype=np.float32)
        self.is_apical = np.array(self.is_apical, dtype=np.float32)
        self.appearance_month = np.array(self.appearance_month, dtype=np.float32)
        self.cycle = np.array(self.cycle, dtype=np.float32)

        self.distance = csgraph.shortest_path(csgraph.csgraph_from_dense(self.adjacency)).astype(np.float32)
        self.nb_descendants = np.count_nonzero(~np.isinf(self.distance) & (self.distance > 0.), axis=1)

        self.appearance_date = np.full(self.GU.shape, np.datetime64('NAT'), dtype='datetime64[D]')
        self.parent_is_apical = np.full(self.GU.shape, 1., dtype=np.float32)
        self.ancestor = np.full(self.GU.shape, 0., dtype=np.float32)
        self.parent_is_apical[np.argwhere(self.adjacency)[:, 1]] = self.is_apical[np.argwhere(self.adjacency)[:, 0]]

        self.bursted = np.zeros(self.GU.shape, dtype=np.float32)
        self.appeared = np.zeros(self.GU.shape, dtype=np.float32)

        self.lsystem = lpy.Lsystem(str(pathlib.Path(__file__).parent.joinpath('topology.lpy')), {
            'process': self,
            'derivation_length': int(nsteps)
        })

        self.lstring = self.lsystem.derive(self.lsystem.axiom, 0, int(np.max(self.distance[np.isfinite(self.distance)])))
        self.is_initially_terminal = (self.cycle == np.nanmax(self.cycle)) & (self.nb_descendants == 0.)

    @xs.runtime(args=('step', 'step_start', 'nsteps'))
    def run_step(self, step, step_start, nsteps):

        # set as property of process, needed by lpy
        self.step = step
        self.step_start = step_start
        self.nsteps = nsteps

        self.bursted[:] = 0.
        self.appeared[:] = 0.

        self.bursted[self.archdev[('arch_dev', 'pot_burst_date')] == step_start] = 1.

        day = step_start.astype('datetime64[D]').item()
        self.current_cycle = self.current_cycle + 1 if day.month == self.month_begin_veg_cycle and day.day == 1 else self.current_cycle

        if np.any(self.bursted):
            total_nb_children = np.sum(self.archdev[('arch_dev', 'pot_nb_lateral_children')][self.bursted == 1.] + self.archdev[('arch_dev', 'pot_has_apical_child')][self.bursted == 1.])
            self.idx_first_child = self.GU.shape[0]
            self.GU = np.append(self.GU, [i + self.GU.shape[0] for i in range(int(total_nb_children))])
            self.GU_ = self.GU
            self.nb_gu = self.GU.shape[0]
            # initialize new GUs
            step_date = step_start.astype('datetime64[D]')
            self.appearance_month[self.idx_first_child:] = step_date.item().month
            self.appearance_date[self.idx_first_child:] = step_date
            self.appeared[self.idx_first_child:] = 1.
            self.adjacency[np.isnan(self.adjacency)] = 0.
            self.cycle[self.idx_first_child:] = self.current_cycle
            self.distance = csgraph.shortest_path(csgraph.csgraph_from_dense(self.adjacency))
            self.lstring = self.lsystem.derive(self.lstring, step, 1)
            self.nb_descendants = np.count_nonzero(~np.isinf(self.distance) & (self.distance > 0.), axis=1)
