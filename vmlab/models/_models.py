import xsimlab as xs

from ..processes import (
    environment,
    fruit_quality,
    fruit_growth,
    light_interception,
    photosynthesis,
    carbon_allocation,
    carbon_balance,
    topology,
    phenology
)


fruit_model = xs.Model({
    'environment': environment.Environment,
    'flower_phenology': phenology.FlowerPhenology,
    'topology': topology.Topology,
    'carbon_allocation': carbon_allocation.CarbonAllocation,
    'light_interception': light_interception.LightInterception,
    'photosynthesis': photosynthesis.Photosythesis,
    'carbon_balance': carbon_balance.CarbonBalance,
    'fruit_growth': fruit_growth.FruitGrowth,
    'fruit_quality': fruit_quality.FruitQuality
})
