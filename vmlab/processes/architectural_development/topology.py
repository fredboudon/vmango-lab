import xsimlab as xs
import numpy as np

from ...enums import Position, Nature


@xs.process
class Topology():

    GU = xs.index(dims='GU')

    rng = xs.any_object(global_name='rng')
    adjacency = xs.variable(dims=('GU', 'GU'), intent='out')
    ancestor = xs.variable(dims='GU', intent='out')
    ancestor_is_apical = xs.variable(dims='GU', intent='out')
    ancestor_nature = xs.variable(dims='GU', intent='out')
    position = xs.variable(dims='GU', intent='out')
    bursted = xs.variable(dims='GU', intent='out')
    appearance_month = xs.variable(dims='GU', intent='out')
    appearance_date = xs.variable(dims='GU', intent='out')
    appeared = xs.variable(dims='GU', intent='out')
    cycle = xs.variable(dims='GU', intent='out')

    current_cycle = xs.variable(intent='inout')
    is_start_of_cycle = xs.variable(dims='month', intent='in')
    seed = xs.variable(intent='in')

    archdev = xs.group_dict('arch_dev')

    def initialize(self):
        self.GU = np.array(['GU0'])
        self.rng = np.random.default_rng(self.seed)
        self.adjacency = np.array([[0.]])
        self.ancestor = np.array([0.])
        self.ancestor_is_apical = np.array([1.])
        self.ancestor_nature = np.array([Nature.VEGETATIVE])
        self.position = np.array([Position.APICAL])
        self.bursted = np.array([0.])
        self.appearance_month = np.array([0.])
        self.appearance_date = np.array(['2002-08-01'], dtype='datetime64[D]')
        self.appeared = np.array([0.])
        self.cycle = np.array([4.])

    @xs.runtime(args=('step', 'step_start'))
    def run_step(self, step, step_start):

        self.bursted = self.archdev[('arch_dev', 'burst_date')] == step_start
        self.appeared[:] = 0.
        self.current_cycle = self.current_cycle + 1 if self.is_start_of_cycle else self.current_cycle

        if np.any(self.bursted):

            nb_lateral_children = self.archdev[('arch_dev', 'nb_lateral_children')]
            has_apical_child = self.archdev[('arch_dev', 'has_apical_child')]
            nature = self.archdev[('arch_dev', 'nature')]

            nb_appeared = np.sum((has_apical_child + nb_lateral_children) * self.bursted)
            nb_gus = self.GU.shape[0]

            self.GU = np.append(self.GU, [f'GU{int(i + nb_gus)}' for i in np.arange(0, nb_appeared)])
            self.appearance_month[nb_gus:] = step_start.astype('datetime64[D]').item().month
            self.appearance_date[nb_gus:] = step_start.astype('datetime64[D]')
            self.appeared[nb_gus:] = 1.

            # initialize new GUs
            self.position[nb_gus:] = Position.LATERAL
            self.adjacency[np.isnan(self.adjacency)] = 0.
            self.cycle[nb_gus:] = self.current_cycle

            gu_child = nb_gus
            for gu_parent in np.flatnonzero(self.bursted):
                for child in np.arange(nb_lateral_children[gu_parent] + has_apical_child[gu_parent], dtype=np.int):
                    if child == 0 and has_apical_child[gu_parent]:
                        self.position[gu_child] = Position.APICAL
                    if self.is_start_of_cycle:
                        self.ancestor[gu_child] = gu_parent
                        self.ancestor_is_apical[gu_child] = self.position[gu_parent]
                        self.ancestor_nature[gu_child] = nature[gu_parent]
                    else:
                        self.ancestor[gu_child] = self.ancestor[gu_parent]
                        self.ancestor_is_apical[gu_child] = self.ancestor_is_apical[gu_parent]
                        self.ancestor_nature[gu_child] = self.ancestor_nature[gu_parent]
                    self.adjacency[gu_parent, gu_child] = 1.

                    gu_child = gu_child + 1
        else:
            self.appeared[:] = 0.

    # def finalize(self):
    #     nb_lateral_children = self.archdev[('other', 'nb_lateral_children')]
    #     has_apical_child = self.archdev[('other', 'has_apical_child')]
    #     nb_appeared = np.sum((has_apical_child + nb_lateral_children) * self.bursted)
    #     nb_gus = self.GU.shape[0]

    #     self.GU = np.append(self.GU, [f'GU{int(i + nb_gus)}' for i in np.arange(0, nb_appeared)])
    #     self.appearance_month[nb_gus:] = 0
    #     self.appeared[nb_gus:] = 1.

    #     # initialize new GUs
    #     self.position[nb_gus:] = Position.LATERAL
    #     self.adjacency[np.isnan(self.adjacency)] = 0.
    #     self.nb_fruits[nb_gus:] = 0.
    #     self.cycle[nb_gus:] = self.current_cycle

    #     gu_child = nb_gus
    #     for gu_parent in np.flatnonzero(self.bursted):
    #         for child in np.arange(nb_lateral_children[gu_parent] + has_apical_child[gu_parent], dtype=np.int):
    #             if child == 0 and has_apical_child[gu_parent]:
    #                 self.position[gu_child] = Position.APICAL
    #             if self.is_start_of_cycle:
    #                 self.ancestor[gu_child + child] = gu_parent
    #                 self.ancestor_is_apical[gu_child + child] = self.position[gu_parent]
    #             else:
    #                 self.ancestor[gu_child + child] = self.ancestor[gu_parent]
    #                 self.ancestor_is_apical[gu_child + child] = self.ancestor_is_apical[gu_parent]
    #             self.adjacency[gu_parent, gu_child + child] = 1.
    #         gu_child = gu_child + child + 1
