import xsimlab as xs

from ..processes import parameters
from ..processes import environment
# from ..processes import architecture
from ..processes import carbon_balance
from ..processes import fruit_growth
from ..processes import growth_unit_growth
from ..processes import inflo_growth
from ..processes import fruit_quality
from ..processes import light_interception
from ..processes import photosynthesis
from ..processes import carbon_unit


fruit_model = xs.Model({
    'params': parameters.Parameters,
    'env': environment.Environment,
    'fruit_growth': fruit_growth.FruitGrowth,
    'inflo_growth': inflo_growth.InfloGrowth,
    'gu_growth': growth_unit_growth.GrowthUnitGrowth,
    'carbon_unit': carbon_unit.Identity,
    'light': light_interception.LightInterception,
    'photo': photosynthesis.Photosythesis,
    'carbon': carbon_balance.CarbonBalance,
    'fruit_quality': fruit_quality.FruitQuality
})
