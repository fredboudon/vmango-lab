import xsimlab as xs

from vmlab.processes.fruit import (
    environment,
    phenology,
    topology,
    light_interception,
    photosynthesis,
    carbon_balance,
    fruit_growth,
    fruit_quality,
)


fruit = xs.Model({
    'environment': environment.Environment,
    'flower_phenology': phenology.FlowerPhenology,
    'topology': topology.Topology,
    'light_interception': light_interception.LightInterception,
    'photosynthesis': photosynthesis.Photosythesis,
    'carbon_balance': carbon_balance.CarbonBalance,
    'fruit_growth': fruit_growth.FruitGrowth,
    'fruit_quality': fruit_quality.FruitQuality
})
