import xsimlab as xs
import numpy as np

from vmlab.constants import (
    R, density_W, MM_water
)

from . import (
    environment,
    carbon_demand,
    carbon_balance,
    phenology,
    fruit_composition
)
from ._base.parameter import ParameterizedProcess


@xs.process
class FruitQuality(ParameterizedProcess):
    """
    Process computing fresh matter and quality (sugars and acids content) changes
    """
    nb_gu = xs.global_ref('nb_gu')

    TM_day = xs.foreign(environment.Environment, 'TM_day')
    GR_day = xs.foreign(environment.Environment, 'GR_day')
    RH_day = xs.foreign(environment.Environment, 'RH_day')

    nmol_solutes = xs.foreign(fruit_composition.FruitComposition, 'nmol_solutes')
    mass_suc = xs.foreign(fruit_composition.FruitComposition, 'mass_suc')
    mass_glc = xs.foreign(fruit_composition.FruitComposition, 'mass_glc')
    mass_frc = xs.foreign(fruit_composition.FruitComposition, 'mass_frc')
    mass_sta = xs.foreign(fruit_composition.FruitComposition, 'mass_sta')
    mass_mal = xs.foreign(fruit_composition.FruitComposition, 'mass_mal')
    mass_cit = xs.foreign(fruit_composition.FruitComposition, 'mass_cit')

    fruit_growth_tts = xs.foreign(phenology.Phenology, 'fruit_growth_tts')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')
    fruited = xs.foreign(phenology.Phenology, 'fruited')

    DM_fruit = xs.foreign(carbon_balance.CarbonBalance, 'DM_fruit')
    DM_fleshpeel_delta = xs.foreign(carbon_balance.CarbonBalance, 'DM_fleshpeel_delta')
    DM_fleshpeel = xs.foreign(carbon_balance.CarbonBalance, 'DM_fleshpeel')
    DM_flesh = xs.foreign(carbon_balance.CarbonBalance, 'DM_flesh')
    DM_fruit_0 = xs.foreign(carbon_demand.CarbonDemand, 'DM_fruit_0')

    FM_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='flesh and peel water mass of the previous day',
        attrs={
            'unit': 'g FM'
        }
    )

    W_fleshpeel = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit flesh and peel water mass',
        attrs={
            'unit': 'g H2O'
        }
    )

    W_flesh = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit flesh water mass',
        attrs={
            'unit': 'g H2O'
        }
    )

    water_potential_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='water potential of the the fruit',
        attrs={
            'unit': 'MPa'
        }
    )

    turgor_pressure_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='turgor pressure in the fruit',
        attrs={
            'unit': 'MPa'
        }
    )

    osmotic_pressure_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='osmotic pressure in the fruit',
        attrs={
            'unit': 'MPa'
        }
    )

    flux_xylem_phloem = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily rate of water inflow in fruit flesh from xylem and phloem',
        attrs={
            'unit': 'g H20 day-1'
        }
    )

    transpiration_fruit = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily rate of fruit transpiration',
        attrs={
            'unit': 'g H20 day-1'
        }
    )

    sucrose = xs.variable(
        dims=('GU'),
        intent='out',
        description='sucrose content in the fruit flesh',
        attrs={
            'unit': 'g g-1 FM'
        },
        groups='fruit_fresh_matter'
    )

    glucose = xs.variable(
        dims=('GU'),
        intent='out',
        description='glucose content in the fruit flesh',
        attrs={
            'unit': 'g g-1 FM'
        }
    )

    fructose = xs.variable(
        dims=('GU'),
        intent='out',
        description='fructose content in the fruit flesh',
        attrs={
            'unit': 'g g-1 FM'
        }
    )

    soluble_sugars = xs.variable(
        dims=('GU'),
        intent='out',
        description='soluble sugar (sucrose, fructuse, glucose) content in the fruit flesh',
        attrs={
            'unit': 'g g-1 FM'
        }
    )

    starch = xs.variable(
        dims=('GU'),
        intent='out',
        description='starch content in the fruit flesh',
        attrs={
            'unit': 'g g-1 FM'
        }
    )

    organic_acids = xs.variable(
        dims=('GU'),
        intent='out',
        description='organic acid (malic and citric acids) content in the fruit flesh',
        attrs={
            'unit': 'g g-1 FM'
        }
    )

    def initialize(self):

        super(FruitQuality, self).initialize()

        self.FM_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.W_fleshpeel = np.zeros(self.nb_gu, dtype=np.float32)
        self.W_flesh = np.zeros(self.nb_gu, dtype=np.float32)
        self.water_potential_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.turgor_pressure_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.osmotic_pressure_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.flux_xylem_phloem = np.zeros(self.nb_gu, dtype=np.float32)
        self.transpiration_fruit = np.zeros(self.nb_gu, dtype=np.float32)
        self.sucrose = np.zeros(self.nb_gu, dtype=np.float32)
        self.glucose = np.zeros(self.nb_gu, dtype=np.float32)
        self.fructose = np.zeros(self.nb_gu, dtype=np.float32)
        self.soluble_sugars = np.zeros(self.nb_gu, dtype=np.float32)
        self.starch = np.zeros(self.nb_gu, dtype=np.float32)
        self.organic_acids = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):

        if np.any(self.nb_fruit > 0.):

            fruiting = np.flatnonzero(self.nb_fruit > 0.)
            params = self.parameters

            h = params.h
            phi_max = params.phi_max
            tau = params.tau
            aLf = params.aLf
            osmotic_pressure_aa = params.osmotic_pressure_aa
            ro = params.ro
            RH_fruit = params.RH_fruit
            Y_0 = params.Y_0
            V_0 = params.V_0
            psat_1 = params.psat_1
            psat_2 = params.psat_2
            density_DM = params.density_DM
            e_fleshpeel2fleshW = params.e_fleshpeel2fleshW
            e_fruitFM2surface_1 = params.e_fruitFM2surface_1
            e_fruitFM2surface_2 = params.e_fruitFM2surface_2
            e_flesh2stoneFM = params.e_flesh2stoneFM
            swp_1 = params.swp_1
            swp_2 = params.swp_2
            swp_3 = params.swp_3
            swp_4 = params.swp_4
            ddthres_1 = params.ddthres_1
            ddthres_2 = params.ddthres_2
            dd_thresh = params.dd_thresh

            e_fruitDM2FM_1 = params.e_fruitDM2FM_1
            e_fruitDM2FM_2 = params.e_fruitDM2FM_2
            e_fruit2fleshW_1 = params.e_fruit2fleshW_1
            e_fruit2fleshW_2 = params.e_fruit2fleshW_2
            e_fruit2peelW_1 = params.e_fruit2peelW_1
            e_fruit2peelW_2 = params.e_fruit2peelW_2

            # initial fresh mass of fruit compartements :
            # from empirical relationships in Léchaudel (2004)

            if np.any(self.fruited):
                fruited = np.flatnonzero(self.fruited)

                self.FM_fruit[fruited] = e_fruitDM2FM_1 * self.DM_fruit[fruited] ** e_fruitDM2FM_2
                self.W_fleshpeel[fruited] = (e_fruit2fleshW_1 * (self.FM_fruit[fruited] - self.DM_fruit[fruited]) ** e_fruit2fleshW_2) + (e_fruit2peelW_1 * (self.FM_fruit[fruited] - self.DM_fruit[fruited]) ** e_fruit2peelW_2)
                self.W_flesh[fruited] = e_fleshpeel2fleshW * self.W_fleshpeel[fruited]

            # ========================================================================================================================
            # OSMOTIC PRESSURE IN THE FRUIT
            # ========================================================================================================================
            # from fruit growth model in Léchaudel et al (2007)

            # -- osmotic pressure in fruit flesh (eq.6-7) :
            self.osmotic_pressure_fruit[fruiting] = (R * (self.TM_day + 273.15) * self.nmol_solutes[fruiting]) / (self.W_flesh[fruiting] / density_W) + osmotic_pressure_aa

            # ========================================================================================================================
            # FRUIT TRANSPIRATION
            # ========================================================================================================================
            # from fruit growth model in Léchaudel et al (2007)

            # -- fruit surface (eq.3) :
            A_fruit = e_fruitFM2surface_1 * self.FM_fruit[fruiting] ** e_fruitFM2surface_2

            # -- saturation vapor pressure (eq.3 in Fishman and Génard 1998) :
            #    converted from bar to MPa
            P_sat = psat_1 * np.exp(psat_2 * self.TM_day) / 10

            # -- fruit transpiration_fruit (eq.2) :
            alpha = MM_water * P_sat / (R * (self.TM_day + 273.15))
            self.transpiration_fruit[fruiting] = A_fruit * alpha * ro * (RH_fruit - self.RH_day / 100)

            # ========================================================================================================================
            # CELL WALL PROPERTIES OF THE FRUIT
            # ========================================================================================================================
            # from fruit growth model in Léchaudel et al (2007)

            # -- cell wall extensibility (eq.18) :
            if np.isnan(dd_thresh):                                                                                                       # _MODIF 2017-05_1
                # if not fixed as input, set from an empirical relationship
                dd_thresh = ddthres_1 * self.DM_fruit_0 + ddthres_2

            Phi = phi_max * tau ** np.maximum(0., self.fruit_growth_tts[fruiting] - dd_thresh)

            # -- threshold pressure (eq.15-16) :
            V = self.W_fleshpeel[fruiting] / density_W + self.DM_fleshpeel[fruiting] / density_DM
            Y = Y_0 + h * (V - V_0)

            # ========================================================================================================================
            # TURGOR PRESSURE & WATER POTENTIAL IN THE FRUIT
            # ========================================================================================================================
            # from fruit growth model in Léchaudel et al (2007)

            # -- water potential of the stem :
            water_potential_stem = swp_1 + swp_2 * self.TM_day + swp_3 * self.RH_day + swp_4 * self.GR_day

            # -- turgor pressure in the fruit (defined by combining eq.11 and eq.13) :
            ALf = A_fruit * aLf
            numerator = Phi * V * Y + ALf * (water_potential_stem + self.osmotic_pressure_fruit[fruiting]) / density_W - self.transpiration_fruit[fruiting] / density_W + self.DM_fleshpeel_delta[fruiting] / density_DM
            denominator = Phi * V + ALf / density_W
            self.turgor_pressure_fruit[fruiting] = numerator / denominator

            self.turgor_pressure_fruit[fruiting] = np.where(
                self.turgor_pressure_fruit[fruiting] < Y,
                np.maximum(Y_0, water_potential_stem + self.osmotic_pressure_fruit[fruiting] - (self.transpiration_fruit[fruiting] - self.DM_fleshpeel_delta[fruiting] * density_W / density_DM) / ALf),
                np.maximum(Y_0, self.turgor_pressure_fruit[fruiting])
            )

            # -- water potential in the fruit (eq.5) :
            self.water_potential_fruit[fruiting] = self.turgor_pressure_fruit[fruiting] - self.osmotic_pressure_fruit[fruiting]

            # ========================================================================================================================
            # WATER AND DRY MATTER CHANGES IN FRUIT COMPARTMENTS
            # ========================================================================================================================
            # from fruit growth model in Léchaudel et al (2007)

            # -- rate of water inflow in the fruit from xylem and phloem (eq.4) :
            self.flux_xylem_phloem[fruiting] = ALf * (water_potential_stem - self.water_potential_fruit[fruiting])

            # -- changes in dry mass, fresh mass and water mass of fruit compartments :
            FM_minus_stone = self.W_flesh[fruiting] + self.DM_flesh[fruiting]
            self.W_fleshpeel[fruiting] = self.W_fleshpeel[fruiting] + self.flux_xylem_phloem[fruiting] - self.transpiration_fruit[fruiting]
            FM_stone = e_flesh2stoneFM * (self.DM_fleshpeel[fruiting] + self.W_fleshpeel[fruiting])
            self.FM_fruit[fruiting] = self.DM_fleshpeel[fruiting] + self.W_fleshpeel[fruiting] + FM_stone
            self.W_flesh[fruiting] = self.W_fleshpeel[fruiting] * e_fleshpeel2fleshW

            self.sucrose[fruiting] = self.mass_suc[fruiting] / FM_minus_stone
            self.glucose[fruiting] = self.mass_glc[fruiting] / FM_minus_stone
            self.fructose[fruiting] = self.mass_frc[fruiting] / FM_minus_stone
            self.soluble_sugars[fruiting] = (self.mass_suc[fruiting] + self.mass_glc[fruiting] + self.mass_frc[fruiting]) / FM_minus_stone
            self.starch[fruiting] = self.mass_sta[fruiting] / FM_minus_stone
            self.organic_acids[fruiting] = (self.mass_mal[fruiting] + self.mass_cit[fruiting]) / FM_minus_stone
