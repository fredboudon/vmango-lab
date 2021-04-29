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
    arch_dev,
    light_interception,
    photosynthesis,
    carbon_allocation,
    carbon_reserve,
    carbon_demand,
    carbon_balance,
    fruit_quality,
    fruit_fresh_matter,
    harvest
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
    'arch_dev': arch_dev.ArchDevStochastic,
    'light_interception': light_interception.LightInterception,
    'photosynthesis': photosynthesis.Photosythesis,
    'carbon_allocation': carbon_allocation.CarbonAllocation,
    'carbon_reserve': carbon_reserve.CarbonReserve,
    'carbon_demand': carbon_demand.CarbonDemand,
    'carbon_balance': carbon_balance.CarbonBalance,
    'fruit_quality': fruit_quality.FruitQuality,
    'fruit_fresh_matter': fruit_fresh_matter.FruitFreshMatter,
    'harvest': harvest.Harvest
})

arch_dev_model = xs.Model({
    'topology': topology.Topology,
    'arch_dev_veg_within': arch_dev_veg_within.ArchDevVegWithin,
    'arch_dev_veg_between': arch_dev_veg_between.ArchDevVegBetween,
    'arch_dev_rep': arch_dev_rep.ArchDevRep,
    'arch_dev_mix': arch_dev_mix.ArchDevMix,
    'arch_dev': arch_dev.ArchDevStochastic
})

fruit_model = xs.Model({
    'environment': environment.Environment,
    'phenology': phenology.Phenology,
    'topology': topology.Topology,
    'geometry': geometry.Geometry,
    'appearance': appearance.Appearance,
    'growth': growth.Growth,
    'arch_dev': arch_dev.ArchDev,
    'light_interception': light_interception.LightInterception,
    'photosynthesis': photosynthesis.Photosythesis,
    'carbon_allocation': carbon_allocation.CarbonAllocation,
    'carbon_reserve': carbon_reserve.CarbonReserve,
    'carbon_demand': carbon_demand.CarbonDemand,
    'carbon_balance': carbon_balance.CarbonBalance,
    'fruit_quality': fruit_quality.FruitQuality,
    'fruit_fresh_matter': fruit_fresh_matter.FruitFreshMatter,
    'harvest': harvest.Harvest
})