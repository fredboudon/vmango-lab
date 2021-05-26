import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib

from ._base.parameter import ParameterizedProcess


@xs.process
class Environment(ParameterizedProcess):
    """
    Reads SMARTIS weather data https://smartis.re/METEOR from csv
    """

    hour = xs.index(dims=('hour'))

    weather_daily_df = None
    weather_hourly_df = None

    weather_file = xs.variable(
        description='path to file with weather data',
        default='',
        static=True
    )

    TM = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly temperature of the ambient atmosphere of the current day',
        attrs={
            'unit': '°C'
        }
    )

    TM_day = xs.variable(
        intent='out',
        description='daily mean temperature of the ambient atmosphere of the current day',
        attrs={
            'unit': '°C'
        }
    )

    GR = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly global radiations of the current day',
        attrs={
            'unit': 'J/cm2/h'
        }
    )

    GR_day = xs.variable(
        intent='out',
        description='mean global radiations of the current day',
        attrs={
            'unit': 'J/cm2/h'
        }
    )

    RH = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly relative humidity of the ambient atmosphere of the current day',
        attrs={
            'unit': '%'
        }
    )

    RH_day = xs.variable(
        intent='out',
        description='mean relative humidity of the ambient atmosphere of the current day',
        attrs={
            'unit': '%'
        }
    )

    def initialize(self):

        super(Environment, self).initialize()

        self.hour = np.arange(24, dtype=np.int8)

        # try to read weather file path from parameter file if not provided in 'input_vars' dict
        if not self.weather_file:
            weather_file_path = pathlib.Path(self.parameter_file_path).parent.joinpath(self.parameters.weather_file_path)
        else:
            weather_file_path = self.weather_file

        self.weather_hourly_df = pd.read_csv(
            weather_file_path,
            sep=';',
            parse_dates=['Jour'],
            dayfirst=True,
            usecols=['Jour', 'tm', 'glot', 'um']
        ).rename(
            columns={'Jour': 'DATETIME', 'tm': 'TM', 'glot': 'GR', 'um': 'RH'},
            inplace=False
        ).set_index('DATETIME', inplace=False).astype(np.float32)

        # smartis may have nans
        self.weather_hourly_df.fillna(inplace=True, method='backfill')

        self.weather_daily_df = pd.DataFrame({
            'TM': self.weather_hourly_df['TM'].groupby(pd.Grouper(freq="1D")).mean(),
            'GR': self.weather_hourly_df['GR'].groupby(pd.Grouper(freq="1D")).mean(),
            'RH': self.weather_hourly_df['RH'].groupby(pd.Grouper(freq="1D")).mean()
        }).astype(np.float32)

    @xs.runtime(args=('step', 'step_start', 'step_end', 'step_delta'))
    def run_step(self, step, step_start, step_end, step_delta):

        step_data = (self.weather_hourly_df.index >= step_end) & (self.weather_hourly_df.index < step_start)

        hourly = self.weather_hourly_df[step_data]

        # SMARTIS data sometimes not complete. Need a strategy to handle missing data
        self.TM = np.resize(hourly['TM'].to_numpy(), 24)
        self.GR = np.resize(hourly['GR'].to_numpy(), 24)
        self.RH = np.resize(hourly['RH'].to_numpy(), 24)
        self.TM_day = self.weather_daily_df['TM'][step_start]
        self.GR_day = self.weather_daily_df['GR'][step_start]
        self.RH_day = self.weather_daily_df['RH'][step_start]

        assert not (np.any(np.isnan(self.TM)) or np.any(np.isnan(self.GR)) or np.any(np.isnan(self.RH)))
