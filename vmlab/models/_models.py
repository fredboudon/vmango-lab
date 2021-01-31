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
    phenology,
    growth_unit_burst
)


fruit_model = xs.Model({
    'params': parameters.Parameters,
    'env': environment.Environment,
    'topo': topology.Topology,
    'pheno_flower': phenology.FlowerPhenology,
    'pheno_gu': phenology.GrowthUnitPhenology,
    'pheno_leaf': phenology.LeafPhenology,
    'fruit_growth': fruit_growth.FruitGrowth,
    'inflo_growth': inflo_growth.InfloGrowth,
    'gu_growth': growth_unit_growth.GrowthUnitGrowth,
    'gu_burst': growth_unit_burst.GrowthUnitBurst,
    'carbon_unit': carbon_unit.Identity,
    'light': light_interception.LightInterception,
    'photo': photosynthesis.Photosythesis,
    'carbon': carbon_balance.CarbonBalance,
    'fruit_quality': fruit_quality.FruitQuality
})
