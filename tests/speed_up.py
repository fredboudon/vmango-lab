import igraph as ig
import numpy as np
import pandas as pd
import gc
import multiprocessing as mp
from timeit import time
import vmlab

from vmlab.models import vmango

# compute speedup ratio for parallel multi-process vs. single, sequential processing

graph = ig.Graph.Tree(250, 2, mode=ig.TREE_OUT)

# make first child of each vertex apical
graph.vs.set_attribute_values('topology__is_apical', 0)
for v in graph.vs:
    successors = v.successors()
    if len(successors):
        successors[0].update_attributes({'topology__is_apical': 1})

tree = vmlab.to_dataframe(graph)


def run(nb_proc, tree):
    setup = vmlab.create_setup(
        model=vmango,
        tree=tree,
        start_date='2003-06-01',
        end_date='2005-06-01',
        setup_toml='vmango.toml',
        input_vars={'geometry__interpretation_freq': 0}
    )
    gc.collect()
    t0 = time.time()
    for _ in range(nb_proc):
        vmlab.run(
            setup,
            vmango,
            progress=False
        )
    took = round(time.time() - t0)
    print('done', nb_proc, took)
    return took


def run_parallel(nb_proc, tree):
    setup = vmlab.create_setup(
        model=vmango,
        tree=tree,
        start_date='2003-06-01',
        end_date='2005-06-01',
        setup_toml='vmango.toml',
        input_vars={'geometry__interpretation_freq': 0}
    )
    gc.collect()
    t0 = time.time()
    vmlab.run(
        setup,
        vmango,
        batch=('test', [{'geometry__interpretation_freq': 0} for _ in range(nb_proc)]),
        progress=False,
        nb_proc=nb_proc
    )
    took = round(time.time() - t0)
    print('done', nb_proc, took)
    return took


times_parallel = np.array([run_parallel(i, tree) for i in range(2, mp.cpu_count() + 1)])
times_single = np.array([run(i, tree) for i in range(2, mp.cpu_count() + 1)])
speed_up = times_single / times_parallel

pd.DataFrame({
    'speed_up': speed_up
}, index=[i for i in range(2, mp.cpu_count() + 1)]).to_csv('speed_up.csv')
