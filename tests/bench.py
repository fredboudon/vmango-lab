import igraph as ig
import numpy as np
import pandas as pd
import xsimlab as xs
import gc
from timeit import time
import vmlab

from vmlab.models import vmango

# compare total time of simulation with various scenarios
# - parallel vs sequential
# - with and without geomety

graphs = [ig.Graph.Tree(int(n), 2, mode=ig.TREE_OUT) for n in np.arange(100, 501, 100)]

# make first child of each vertex apical
for graph in graphs:
    graph.vs.set_attribute_values('topology__is_apical', 0)
    for v in graph.vs:
        successors = v.successors()
        if len(successors):
            successors[0].update_attributes({'topology__is_apical': 1})

trees = [vmlab.to_dataframe(graph) for graph in graphs]

nb_gus = []


@xs.runtime_hook(stage='finalize')
def hook(model, context, state):
    nb = state[('topology', 'is_apical')].shape[0]
    if nb not in nb_gus:
        nb_gus.append(nb)


def run(i, tree, freq):
    setup = vmlab.create_setup(
        model=vmango,
        tree=tree,
        start_date='2003-06-01',
        end_date='2005-06-01',
        setup_toml='vmango.toml',
        input_vars={'geometry__interpretation_freq': freq},
        output_vars={'topology__is_apical': None}
    )
    gc.collect()
    t0 = time.time()
    for _ in range(4):
        vmlab.run(
            setup,
            vmango,
            hooks=[hook],
            progress=False
        )
    took = round(time.time() - t0)
    print('done', i, took)
    return took


def run_parallel(i, tree, freq):
    setup = vmlab.create_setup(
        model=vmango,
        tree=tree,
        start_date='2003-06-01',
        end_date='2005-06-01',
        setup_toml='vmango.toml',
        input_vars={'geometry__interpretation_freq': freq},
        output_vars={'topology__is_apical': None}
    )
    gc.collect()
    t0 = time.time()
    vmlab.run(
        setup,
        vmango,
        batch=('test', [{'geometry__interpretation_freq': freq} for _ in range(4)]),
        progress=False
    )
    took = round(time.time() - t0)
    print('done', i, took)
    return took


times_geo_30_parallel = [run_parallel(i, tree, 30) for i, tree in enumerate(trees)]
times_geo_0_parallel = [run_parallel(i, tree, 0) for i, tree in enumerate(trees)]
times_geo_30 = [run(i, tree, 30) for i, tree in enumerate(trees)]
times_geo_0 = [run(i, tree, 0) for i, tree in enumerate(trees)]

pd.DataFrame({
    'S0': times_geo_0,
    'S30': times_geo_30,
    'P0': times_geo_0_parallel,
    'P30': times_geo_30_parallel
}, index=nb_gus).to_csv('bench.csv')
