import xsimlab as xs
import numpy as np

from vmlab.constants import (
    MM_mal, MM_cit, MM_pyr, MM_oxa, MM_K, MM_Mg, MM_Ca, MM_NH4, MM_Na, MM_glc, MM_frc, MM_suc
)

from . import (
    carbon_balance,
    phenology
)
from ._base.parameter import ParameterizedProcess


@xs.process
class FruitComposition(ParameterizedProcess):
    """
    Process computing fruit composition ( mass and number of moles of soluble sugars, starch, acids and minerals)
    """

    nb_gu = xs.global_ref('nb_gu')

    fruit_growth_tts = xs.foreign(phenology.Phenology, 'fruit_growth_tts')
    nb_fruit = xs.foreign(phenology.Phenology, 'nb_fruit')

    DM_fruit = xs.foreign(carbon_balance.CarbonBalance, 'DM_fruit')
    DM_fleshpeel = xs.foreign(carbon_balance.CarbonBalance, 'DM_fleshpeel')
    DM_flesh = xs.foreign(carbon_balance.CarbonBalance, 'DM_flesh')
    # this dummy just forces process ordering
    __dummy = xs.foreign(carbon_balance.CarbonBalance, 'carbon_supply')

    nmol_solutes = xs.variable(
        dims=('GU'),
        intent='out',
        description='number of moles of osmotically active solutes in fruit flesh',
        attrs={
            'unit': 'mol'
        }
    )

    mass_suc = xs.variable(
        dims=('GU'),
        intent='out',
        description='sucrose mass in the fruit flesh',
        attrs={
            'unit': 'g'
        }
    )

    mass_glc = xs.variable(
        dims=('GU'),
        intent='out',
        description='glucose mass in the fruit flesh',
        attrs={
            'unit': 'g'
        }
    )

    mass_frc = xs.variable(
        dims=('GU'),
        intent='out',
        description='fructose mass in the fruit flesh',
        attrs={
            'unit': 'g'
        }
    )

    mass_sta = xs.variable(
        dims=('GU'),
        intent='out',
        description='starch mass in the fruit flesh',
        attrs={
            'unit': 'g'
        }
    )

    mass_mal = xs.variable(
        dims=('GU'),
        intent='out',
        description='malic acid mass in the fruit flesh',
        attrs={
            'unit': 'g'
        }
     )

    mass_cit = xs.variable(
        dims=('GU'),
        intent='out',
        description='citric acid mass in the fruit flesh',
        attrs={
            'unit': 'g'
        }
    )

    def initialize(self):

        super(FruitComposition, self).initialize()

        self.nmol_solutes = np.zeros(self.nb_gu, dtype=np.float32)
        self.mass_suc = np.zeros(self.nb_gu, dtype=np.float32)
        self.mass_glc = np.zeros(self.nb_gu, dtype=np.float32)
        self.mass_frc = np.zeros(self.nb_gu, dtype=np.float32)
        self.mass_sta = np.zeros(self.nb_gu, dtype=np.float32)
        self.mass_mal = np.zeros(self.nb_gu, dtype=np.float32)
        self.mass_cit = np.zeros(self.nb_gu, dtype=np.float32)

    @xs.runtime(args=())
    def run_step(self):

        if np.any(self.nb_fruit > 0.):

            fruiting = np.flatnonzero(self.nb_fruit > 0.)
            params = self.parameters

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
            Tthresh_fruit_stage = params.Tthresh_fruit_stage

            # ========================================================================================================================
            # MASS AND NUMBER OF MOLES OF OSMOTICALLY ACTIVE SOLUTES IN FRUIT FLESH
            # ========================================================================================================================
            # from empirical relationships in Léchaudel et al (2007)

            # -- mass proportion of osmotically active solutes & starch in the dry mass of fruit flesh (eq.9) :
            fruit_growth_tts_delta = self.fruit_growth_tts[fruiting] - Tthresh_fruit_stage
            DM_flesh_x_tts_delta = self.DM_flesh[fruiting] * fruit_growth_tts_delta
            prop_mal = np.maximum(0, delta_mal[0] + delta_mal[1] * fruit_growth_tts_delta + delta_mal[2] * self.DM_flesh[fruiting] + delta_mal[3] * DM_flesh_x_tts_delta)
            prop_cit = np.maximum(0, delta_cit[0] + delta_cit[1] * fruit_growth_tts_delta + delta_cit[2] * self.DM_flesh[fruiting] + delta_cit[3] * DM_flesh_x_tts_delta)
            prop_pyr = np.maximum(0, delta_pyr[0] + delta_pyr[1] * fruit_growth_tts_delta + delta_pyr[2] * self.DM_flesh[fruiting] + delta_pyr[3] * DM_flesh_x_tts_delta)
            prop_oxa = np.maximum(0, delta_oxa[0] + delta_oxa[1] * fruit_growth_tts_delta + delta_oxa[2] * self.DM_flesh[fruiting] + delta_oxa[3] * DM_flesh_x_tts_delta)
            prop_K = np.maximum(0, delta_K[0] + delta_K[1] * fruit_growth_tts_delta + delta_K[2] * self.DM_flesh[fruiting] + delta_K[3] * DM_flesh_x_tts_delta)
            prop_Mg = np.maximum(0, delta_Mg[0] + delta_Mg[1] * fruit_growth_tts_delta + delta_Mg[2] * self.DM_flesh[fruiting] + delta_Mg[3] * DM_flesh_x_tts_delta)
            prop_Ca = np.maximum(0, delta_Ca[0] + delta_Ca[1] * fruit_growth_tts_delta + delta_Ca[2] * self.DM_flesh[fruiting] + delta_Ca[3] * DM_flesh_x_tts_delta)
            prop_NH4 = np.maximum(0, delta_NH4[0] + delta_NH4[1] * fruit_growth_tts_delta + delta_NH4[2] * self.DM_flesh[fruiting] + delta_NH4[3] * DM_flesh_x_tts_delta)
            prop_Na = np.maximum(0, delta_Na[0] + delta_Na[1] * fruit_growth_tts_delta + delta_Na[2] * self.DM_flesh[fruiting] + delta_Na[3] * DM_flesh_x_tts_delta)
            prop_glc = np.maximum(0, delta_glc[0] + delta_glc[1] * fruit_growth_tts_delta + delta_glc[2] * self.DM_flesh[fruiting] + delta_glc[3] * DM_flesh_x_tts_delta)
            prop_frc = np.maximum(0, delta_frc[0] + delta_frc[1] * fruit_growth_tts_delta + delta_frc[2] * self.DM_flesh[fruiting] + delta_frc[3] * DM_flesh_x_tts_delta)
            prop_suc = np.maximum(0, delta_suc[0] + delta_suc[1] * fruit_growth_tts_delta + delta_suc[2] * self.DM_flesh[fruiting] + delta_suc[3] * DM_flesh_x_tts_delta)
            prop_sta = np.maximum(0, delta_sta[0] + delta_sta[1] * fruit_growth_tts_delta + delta_sta[2] * self.DM_flesh[fruiting] + delta_sta[3] * DM_flesh_x_tts_delta)

            # -- mass and number of moles of osmotically active solutes & starch in fruit flesh (eq.8) :
            self.mass_mal[fruiting] = prop_mal * self.DM_flesh[fruiting]
            nmol_mal = self.mass_mal[fruiting] / MM_mal
            self.mass_cit[fruiting] = prop_cit * self.DM_flesh[fruiting]
            nmol_cit = self.mass_cit[fruiting] / MM_cit
            mass_pyr = prop_pyr * self.DM_flesh[fruiting]
            nmol_pyr = mass_pyr / MM_pyr
            mass_oxa = prop_oxa * self.DM_flesh[fruiting]
            nmol_oxa = mass_oxa / MM_oxa
            mass_K = prop_K * self.DM_flesh[fruiting]
            nmol_K = mass_K / MM_K
            mass_Mg = prop_Mg * self.DM_flesh[fruiting]
            nmol_Mg = mass_Mg / MM_Mg
            mass_Ca = prop_Ca * self.DM_flesh[fruiting]
            nmol_Ca = mass_Ca / MM_Ca
            mass_NH4 = prop_NH4 * self.DM_flesh[fruiting]
            nmol_NH4 = mass_NH4 / MM_NH4
            mass_Na = prop_Na * self.DM_flesh[fruiting]
            nmol_Na = mass_Na / MM_Na
            self.mass_glc[fruiting] = prop_glc * self.DM_flesh[fruiting]
            nmol_glc = self.mass_glc[fruiting] / MM_glc
            self.mass_frc[fruiting] = prop_frc * self.DM_flesh[fruiting]
            nmol_frc = self.mass_frc[fruiting] / MM_frc
            self.mass_suc[fruiting] = prop_suc * self.DM_flesh[fruiting]
            nmol_suc = self.mass_suc[fruiting] / MM_suc
            self.mass_sta[fruiting] = prop_sta * self.DM_flesh[fruiting]

            # -- osmotic pressure in fruit flesh (eq.7) :
            self.nmol_solutes[fruiting] = nmol_mal + nmol_cit + nmol_pyr + nmol_oxa + nmol_K + nmol_Mg + nmol_Ca + nmol_NH4 + nmol_Na + nmol_glc + nmol_frc + nmol_suc
