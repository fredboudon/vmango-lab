import xsimlab as xs
import numpy as np

from vmlab.constants import (
    R, density_W, MM_water, MM_mal, MM_cit, MM_pyr, MM_oxa, MM_K,
    MM_Mg, MM_Ca, MM_NH4, MM_Na, MM_glc, MM_frc, MM_suc
)

from . import (
    environment,
    carbon_demand,
    carbon_balance,
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class FruitQuality(ParameterizedProcess):

    T_air = xs.foreign(environment.Environment, 'T_air')
    GR = xs.foreign(environment.Environment, 'GR')
    RH = xs.foreign(environment.Environment, 'RH')
    TM_air = xs.foreign(environment.Environment, 'TM_air')

    fruit_growth_tts_delta = xs.foreign(phenology.Phenology, 'fruit_growth_tts_delta')

    DM_fruit_0 = xs.foreign(carbon_demand.CarbonDemand, 'DM_fruit_0')

    nb_gu = xs.global_ref('nb_gu')

    DM_fruit_delta = xs.foreign(carbon_balance.CarbonBalance, 'DM_fruit_delta')
    DM_fruit = xs.foreign(carbon_balance.CarbonBalance, 'DM_fruit')

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

    DM_fleshpeel = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit flesh and peel dry mass',
        attrs={
            'unit': 'g DM'
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

    DM_flesh = xs.variable(
        dims=('GU'),
        intent='out',
        description='fruit flesh dry mass',
        attrs={
            'unit': 'g DM'
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
        }
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

        self.FM_fruit = np.zeros(self.nb_gu)
        self.W_fleshpeel = np.zeros(self.nb_gu)
        self.DM_fleshpeel = np.zeros(self.nb_gu)
        self.W_flesh = np.zeros(self.nb_gu)
        self.DM_flesh = np.zeros(self.nb_gu)
        self.water_potential_fruit = np.zeros(self.nb_gu)
        self.turgor_pressure_fruit = np.zeros(self.nb_gu)
        self.osmotic_pressure_fruit = np.zeros(self.nb_gu)
        self.flux_xylem_phloem = np.zeros(self.nb_gu)
        self.transpiration_fruit = np.zeros(self.nb_gu)
        self.sucrose = np.zeros(self.nb_gu)
        self.glucose = np.zeros(self.nb_gu)
        self.fructose = np.zeros(self.nb_gu)
        self.soluble_sugars = np.zeros(self.nb_gu)
        self.starch = np.zeros(self.nb_gu)
        self.organic_acids = np.zeros(self.nb_gu)
        self.ripe = np.zeros(self.nb_gu)

    @xs.runtime(args=())
    def run_step(self):

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
        e_fruit2peelDM_1 = params.e_fruit2peelDM_1
        e_fruit2peelDM_2 = params.e_fruit2peelDM_2
        e_fruit2fleshDM_1 = params.e_fruit2fleshDM_1
        e_fruit2fleshDM_2 = params.e_fruit2fleshDM_2
        e_fleshpeel2fleshDM = params.e_fleshpeel2fleshDM
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
        delta_mal = params.delta.mal
        delta_cit = params.delta.cit
        delta_pyr = params.delta.pyr
        delta_oxa = params.delta.oxa
        delta_K = params.delta.K
        delta_Mg = params.delta.Mg
        delta_Ca = params.delta.Ca
        delta_NH4 = params.delta.NH4
        delta_Na = params.delta.Na
        delta_glc = params.delta.glc
        delta_frc = params.delta.frc
        delta_suc = params.delta.suc
        delta_sta = params.delta.sta
        dd_thresh = params.dd_thresh

        e_fruitDM2FM_1 = params.e_fruitDM2FM_1
        e_fruitDM2FM_2 = params.e_fruitDM2FM_2
        e_fruit2fleshW_1 = params.e_fruit2fleshW_1
        e_fruit2fleshW_2 = params.e_fruit2fleshW_2
        e_fruit2peelW_1 = params.e_fruit2peelW_1
        e_fruit2peelW_2 = params.e_fruit2peelW_2

        # initial fresh and dry mass of fruit compartements :
        # from empirical relationships in Léchaudel (2004)

        self.FM_fruit = np.array([e_fruitDM2FM_1 * DM_fruit ** e_fruitDM2FM_2
                                  if DM_fruit > 0 and FM_fruit == 0 else 0. if DM_fruit == 0 else FM_fruit
                                  for DM_fruit, FM_fruit in zip(self.DM_fruit, self.FM_fruit)])

        self.W_fleshpeel = np.array([(e_fruit2fleshW_1 * (FM_fruit - DM_fruit) ** e_fruit2fleshW_2) + (e_fruit2peelW_1 * (FM_fruit - DM_fruit) ** e_fruit2peelW_2)
                                     if FM_fruit > 0 and W_fleshpeel == 0 else 0. if FM_fruit == 0 else W_fleshpeel
                                     for DM_fruit, FM_fruit, W_fleshpeel in zip(self.DM_fruit, self.FM_fruit, self.W_fleshpeel)])

        self.DM_fleshpeel = np.array([(e_fruit2fleshDM_1 * DM_fruit ** e_fruit2fleshDM_2) + (e_fruit2peelDM_1 * DM_fruit ** e_fruit2peelDM_2)
                                      if DM_fruit > 0 and DM_fleshpeel == 0 else 0. if DM_fruit == 0 else DM_fleshpeel
                                      for DM_fruit, DM_fleshpeel in zip(self.DM_fruit, self.DM_fleshpeel)])

        self.W_flesh = np.array([e_fleshpeel2fleshW * W_fleshpeel
                                 if W_fleshpeel > 0 and W_flesh == 0 else 0. if W_fleshpeel == 0 else W_flesh
                                 for W_fleshpeel, W_flesh in zip(self.W_fleshpeel, self.W_flesh)])

        self.DM_flesh = np.ones(self.nb_gu) * e_fleshpeel2fleshDM * self.DM_fleshpeel

        TM_air = np.mean(self.TM_air)
        RH = np.mean(self.RH)
        GR = np.mean(self.GR)

        # ========================================================================================================================
        # DRY MASS AND GROWTH RATE OF FRUIT FLESH
        # ========================================================================================================================
        # from empirical relationships in Léchaudel (2004)

        DM_fruit_previous = self.DM_fruit - self.DM_fruit_delta
        DM_fleshpeel_previous = (e_fruit2fleshDM_1 * (DM_fruit_previous) ** e_fruit2fleshDM_2) + (e_fruit2peelDM_1 * (DM_fruit_previous) ** e_fruit2peelDM_2)
        DM_fleshpeel_delta = np.array([(e_fruit2fleshDM_1 * e_fruit2fleshDM_2 * DM_fruit ** (e_fruit2fleshDM_2 - 1) + e_fruit2peelDM_1 * e_fruit2peelDM_2 *
                                        DM_fruit ** (e_fruit2peelDM_2 - 1)) * DM_fruit_delta if DM_fruit > 0 else 0 for DM_fruit, DM_fruit_delta in zip(self.DM_fruit, self.DM_fruit_delta)])
        DM_fleshpeel_delta = np.maximum(DM_fleshpeel_delta, 0)
        DM_flesh_previous = e_fleshpeel2fleshDM * DM_fleshpeel_previous
        W_flesh_previous = e_fleshpeel2fleshW * self.W_fleshpeel

        # ========================================================================================================================
        # OSMOTIC PRESSURE IN THE FRUIT
        # ========================================================================================================================

        # -- mass proportion of osmotically active solutes & starch in the dry mass of fruit flesh (eq.9) :
        DM_flesh_x_tts_delta = DM_flesh_previous * self.fruit_growth_tts_delta
        prop_mal = np.maximum(0, delta_mal[0] + delta_mal[1] * self.fruit_growth_tts_delta + delta_mal[2] * DM_flesh_previous + delta_mal[3] * DM_flesh_x_tts_delta)
        prop_cit = np.maximum(0, delta_cit[0] + delta_cit[1] * self.fruit_growth_tts_delta + delta_cit[2] * DM_flesh_previous + delta_cit[3] * DM_flesh_x_tts_delta)
        prop_pyr = np.maximum(0, delta_pyr[0] + delta_pyr[1] * self.fruit_growth_tts_delta + delta_pyr[2] * DM_flesh_previous + delta_pyr[3] * DM_flesh_x_tts_delta)
        prop_oxa = np.maximum(0, delta_oxa[0] + delta_oxa[1] * self.fruit_growth_tts_delta + delta_oxa[2] * DM_flesh_previous + delta_oxa[3] * DM_flesh_x_tts_delta)
        prop_K = np.maximum(0, delta_K[0] + delta_K[1] * self.fruit_growth_tts_delta + delta_K[2] * DM_flesh_previous + delta_K[3] * DM_flesh_x_tts_delta)
        prop_Mg = np.maximum(0, delta_Mg[0] + delta_Mg[1] * self.fruit_growth_tts_delta + delta_Mg[2] * DM_flesh_previous + delta_Mg[3] * DM_flesh_x_tts_delta)
        prop_Ca = np.maximum(0, delta_Ca[0] + delta_Ca[1] * self.fruit_growth_tts_delta + delta_Ca[2] * DM_flesh_previous + delta_Ca[3] * DM_flesh_x_tts_delta)
        prop_NH4 = np.maximum(0, delta_NH4[0] + delta_NH4[1] * self.fruit_growth_tts_delta + delta_NH4[2] * DM_flesh_previous + delta_NH4[3] * DM_flesh_x_tts_delta)
        prop_Na = np.maximum(0, delta_Na[0] + delta_Na[1] * self.fruit_growth_tts_delta + delta_Na[2] * DM_flesh_previous + delta_Na[3] * DM_flesh_x_tts_delta)
        prop_glc = np.maximum(0, delta_glc[0] + delta_glc[1] * self.fruit_growth_tts_delta + delta_glc[2] * DM_flesh_previous + delta_glc[3] * DM_flesh_x_tts_delta)
        prop_frc = np.maximum(0, delta_frc[0] + delta_frc[1] * self.fruit_growth_tts_delta + delta_frc[2] * DM_flesh_previous + delta_frc[3] * DM_flesh_x_tts_delta)
        prop_suc = np.maximum(0, delta_suc[0] + delta_suc[1] * self.fruit_growth_tts_delta + delta_suc[2] * DM_flesh_previous + delta_suc[3] * DM_flesh_x_tts_delta)
        prop_sta = np.maximum(0, delta_sta[0] + delta_sta[1] * self.fruit_growth_tts_delta + delta_sta[2] * DM_flesh_previous + delta_sta[3] * DM_flesh_x_tts_delta)

        # -- mass and number of moles of osmotically active solutes & starch in fruit flesh (eq.8) :
        mass_mal = prop_mal * DM_flesh_previous
        nmol_mal = mass_mal / MM_mal
        mass_cit = prop_cit * DM_flesh_previous
        nmol_cit = mass_cit / MM_cit
        mass_pyr = prop_pyr * DM_flesh_previous
        nmol_pyr = mass_pyr / MM_pyr
        mass_oxa = prop_oxa * DM_flesh_previous
        nmol_oxa = mass_oxa / MM_oxa
        mass_K = prop_K * DM_flesh_previous
        nmol_K = mass_K / MM_K
        mass_Mg = prop_Mg * DM_flesh_previous
        nmol_Mg = mass_Mg / MM_Mg
        mass_Ca = prop_Ca * DM_flesh_previous
        nmol_Ca = mass_Ca / MM_Ca
        mass_NH4 = prop_NH4 * DM_flesh_previous
        nmol_NH4 = mass_NH4 / MM_NH4
        mass_Na = prop_Na * DM_flesh_previous
        nmol_Na = mass_Na / MM_Na
        mass_glc = prop_glc * DM_flesh_previous
        nmol_glc = mass_glc / MM_glc
        mass_frc = prop_frc * DM_flesh_previous
        nmol_frc = mass_frc / MM_frc
        mass_suc = prop_suc * DM_flesh_previous
        nmol_suc = mass_suc / MM_suc
        mass_sta = prop_sta * DM_flesh_previous

        # -- osmotic pressure in fruit flesh (eq.6-7) :
        nmol_solutes = nmol_mal + nmol_cit + nmol_pyr + nmol_oxa + nmol_K + nmol_Mg + nmol_Ca + nmol_NH4 + nmol_Na + nmol_glc + nmol_frc + nmol_suc
        self.osmotic_pressure_fruit = np.array([(R * (TM_air + 273.15) * nmol_solutes) / (W_flesh_previous / density_W) + osmotic_pressure_aa
                                                if W_flesh_previous > 0 else 0. for nmol_solutes, W_flesh_previous in zip(nmol_solutes, W_flesh_previous)])

        # ========================================================================================================================
        # FRUIT TRANSPIRATION
        # ========================================================================================================================

        # -- fruit surface (eq.3) :
        A_fruit = e_fruitFM2surface_1 * self.FM_fruit ** e_fruitFM2surface_2

        # -- saturation vapor pressure (eq.3 in Fishman and Génard 1998) :
        #    converted from bar to MPa
        P_sat = psat_1 * np.exp(psat_2 * TM_air) / 10

        # -- fruit transpiration_fruit (eq.2) :
        alpha = MM_water * P_sat / (R * (TM_air + 273.15))
        self.transpiration_fruit = A_fruit * alpha * ro * (RH_fruit - RH / 100)

        # ========================================================================================================================
        # CELL WALL PROPERTIES OF THE FRUIT
        # ========================================================================================================================

        # -- cell wall extensibility (eq.18) :
        if np.isnan(dd_thresh):                                                                                                       # _MODIF 2017-05_1
            # if not fixed as input, set from an empirical relationship
            dd_thresh = ddthres_1 * self.DM_fruit_0 + ddthres_2

        Phi = phi_max * tau ** np.maximum(0., self.fruit_growth_tts_delta - dd_thresh)

        # -- threshold pressure (eq.15-16) :
        V = self.W_fleshpeel / density_W + DM_fleshpeel_previous / density_DM
        Y = Y_0 + h * (V - V_0)

        # ========================================================================================================================
        # TURGOR PRESSURE & WATER POTENTIAL IN THE FRUIT
        # ========================================================================================================================

        # -- water potential of the stem :
        water_potential_stem = swp_1 + swp_2 * TM_air + swp_3 * RH + swp_4 * GR

        # -- turgor pressure in the fruit (defined by combining eq.11 and eq.13) :
        ALf = A_fruit * aLf
        numerator = Phi * V * Y + ALf * (water_potential_stem + self.osmotic_pressure_fruit) / density_W - self.transpiration_fruit / density_W + DM_fleshpeel_delta / density_DM
        denominator = Phi * V + ALf / density_W
        self.turgor_pressure_fruit = np.array([numerator / denominator
                                               if denominator > 0 else 0. for numerator, denominator in zip(numerator, denominator)])

        self.turgor_pressure_fruit = np.array([water_potential_stem + osmotic_pressure_fruit - (transpiration_fruit - DM_fleshpeel_delta * density_W / density_DM) / ALf
                                               if ALf > 0 and turgor_pressure_fruit < Y else 0. if ALf == 0 else turgor_pressure_fruit
                                               for turgor_pressure_fruit, osmotic_pressure_fruit, transpiration_fruit, DM_fleshpeel_delta, ALf, Y in zip(self.turgor_pressure_fruit, self.osmotic_pressure_fruit, self.transpiration_fruit, DM_fleshpeel_delta, ALf, Y)])

        self.turgor_pressure_fruit = np.where(
            self.turgor_pressure_fruit < Y_0,
            0,
            self.turgor_pressure_fruit
        )

        # -- water potential in the fruit (eq.5) :
        self.water_potential_fruit = self.turgor_pressure_fruit - self.osmotic_pressure_fruit

        # ========================================================================================================================
        # WATER AND DRY MATTER CHANGES IN FRUIT COMPARTMENTS
        # ========================================================================================================================

        # -- rate of water inflow in the fruit from xylem and phloem (eq.4) :
        self.flux_xylem_phloem = ALf * (water_potential_stem - self.water_potential_fruit)

        # -- changes in dry mass, fresh mass and water mass of fruit compartments :
        FM_minus_stone = self.W_flesh + self.DM_flesh
        self.DM_fleshpeel = DM_fleshpeel_previous + DM_fleshpeel_delta
        self.W_fleshpeel = self.W_fleshpeel + self.flux_xylem_phloem - self.transpiration_fruit
        FM_stone = e_flesh2stoneFM * (self.DM_fleshpeel + self.W_fleshpeel)
        self.FM_fruit = self.DM_fleshpeel + self.W_fleshpeel + FM_stone
        self.DM_flesh = self.DM_fleshpeel * e_fleshpeel2fleshDM
        self.W_flesh = self.W_fleshpeel * e_fleshpeel2fleshW

        self.sucrose = np.array([mass_suc / FM_minus_stone if FM_minus_stone > 0 else 0.
                                 for FM_minus_stone, mass_suc in zip(FM_minus_stone, mass_suc)])
        self.glucose = np.array([mass_glc / FM_minus_stone if FM_minus_stone > 0 else 0.
                                 for FM_minus_stone, mass_glc in zip(FM_minus_stone, mass_glc)])
        self.fructose = np.array([mass_frc / FM_minus_stone if FM_minus_stone > 0 else 0.
                                 for FM_minus_stone, mass_frc in zip(FM_minus_stone, mass_frc)])
        self.soluble_sugars = np.array([(mass_suc + mass_glc + mass_frc) / FM_minus_stone if FM_minus_stone > 0 else 0.
                                        for FM_minus_stone, mass_suc, mass_glc, mass_frc in zip(FM_minus_stone, mass_suc, mass_glc, mass_frc)])
        self.starch = np.array([mass_sta / FM_minus_stone if FM_minus_stone > 0 else 0.
                                for FM_minus_stone, mass_sta in zip(FM_minus_stone, mass_sta)])
        self.organic_acids = np.array([(mass_mal + mass_cit) / FM_minus_stone if FM_minus_stone > 0 else 0.
                                       for FM_minus_stone, mass_mal, mass_cit in zip(FM_minus_stone, mass_mal, mass_cit)])

    def finalize_step(self):
        pass

    def finalize(self):
        pass