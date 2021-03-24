import xsimlab as xs

from ..processes import (
    topology,
    geometry,
    environment,
    phenology,
    appearance,
    growth,
    arch_dev_veg_within,
    arch_dev_veg_between,
    arch_dev_rep,
    arch_dev_mix,
    arch_dev
)


vmango = xs.Model({
    'environment': environment.Environment,
    'phenology': phenology.Phenology,
    'topology': topology.Topology,
    'geometry': geometry.Geometry,
    'appearance': appearance.Appearance,
    'growth': growth.Growth,
    'arch_dev_veg_within': arch_dev_veg_within.ArchDevVegWithin,
    'arch_dev_veg_between': arch_dev_veg_between.ArchDevVegBetween,
    'arch_dev_rep': arch_dev_rep.ArchDevRep,
    'arch_dev_mix': arch_dev_mix.ArchDevMix,
    'arch_dev': arch_dev.ArchDev
})
