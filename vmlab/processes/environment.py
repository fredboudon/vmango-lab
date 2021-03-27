import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib

from ._base.parameter import BaseParameterizedProcess


@xs.process
class Environment(BaseParameterizedProcess):

    hour = xs.index(dims=('hour'))

    weather_daily_df = None
    weather_hourly_df = None

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

    RH = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly relative humidity of the ambient atmosphere of the current day',
        attrs={
            'unit': '%'
        }
    )

    def initialize(self):

        super(Environment, self).initialize()

        self.hour = np.arange(24, dtype=np.int8)

        weather_file_path = pathlib.Path(self.parameter_file_path).parent.joinpath(self.parameters.weather_file_path)

        smartis = pd.read_csv(
            weather_file_path,
            sep=';',
            parse_dates=['Jour'],
            dayfirst=True,
            usecols=['Jour', 'tm', 'glot', 'um']
        )

        self.weather_hourly_df = smartis.rename(
            columns={'Jour': 'DATETIME', 'tm': 'TM', 'glot': 'GR', 'um': 'RH'},
            inplace=False
        ).set_index('DATETIME', inplace=False).astype(np.float32)

        self.weather_daily_df = pd.DataFrame({
            'TM': self.weather_hourly_df['TM'].groupby(pd.Grouper(freq="1D")).mean()
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
