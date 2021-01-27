import xsimlab as xs

from ..processes import (
    parameters,
    environment,
    fruit_growth,
    growth_unit_growth,
    inflo_growth,
    fruit_quality,
    light_interception,
    photosynthesis,
    carbon_unit,
    carbon_balance,
    topology,
    phenology
)


fruit_model = xs.Model({
    'params': parameters.Parameters,
    'env': environment.Environment,
    'topo': topology.Topology,
    'pheno': phenology.Phenology,
    'fruit_growth': fruit_growth.FruitGrowth,
    'inflo_growth': inflo_growth.InfloGrowth,
    'gu_growth': growth_unit_growth.GrowthUnitGrowth,
    'carbon_unit': carbon_unit.Identity,
    'light': light_interception.LightInterception,
    'photo': photosynthesis.Photosythesis,
    'carbon': carbon_balance.CarbonBalance,
    'fruit_quality': fruit_quality.FruitQuality
})
