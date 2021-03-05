import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib

from vmlab.processes import BaseParameterizedProcess


@xs.process
class Environment(BaseParameterizedProcess):

    hour = xs.index(dims=('hour'))

    weather_df = None
    weather_daily_df = None
    weather_hourly_df = None

    # numpy random generator
    rng = xs.any_object(global_name='rng')

    T_air = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly temperature of the ambient atmosphere of the current day',
        attrs={
            'unit': '°C'
        }
    )

    TM_air = xs.variable(
        dims=('hour'),
        intent='out',
        description='hourly (effectively 24 x daily) temperature of the ambient atmosphere of the current day',
        attrs={
            'unit': '°C'
        }
    )

    TM = xs.variable(
        intent='out',
        description='daily mean temperature of the ambient atmosphere of the current day',
        attrs={
            'unit': '°C'
        }
    )

    T_fruit = xs.variable(
        dims=('hour'),
        intent='out',
        description='temperature of the fruit of the current day',
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

        self.rng = np.random.default_rng(self.parameters.seed)

        folder_path = pathlib.Path(self.parameter_file_path).parent
        weather_hourly_file_path = folder_path.joinpath(self.parameters.weather_hourly_file_path)
        weather_daily_file_path = folder_path.joinpath(self.parameters.weather_daily_file_path)

        weather_hourly_df = pd.read_csv(
            weather_hourly_file_path,
            sep=';',
            parse_dates=['DATETIME'],
            dayfirst=True,
            usecols=['HOUR', 'GR', 'T', 'RH', 'DATETIME'],
            dtype={'GR': np.float, 'T': np.float, 'RH': np.float}
        )

        weather_daily_df = pd.read_csv(
            weather_daily_file_path,
            sep=';',
            parse_dates=['DATE'],
            dayfirst=True,
            usecols=['DATE', 'TM', 'TX', 'TN'],
            dtype={'TM': np.float, 'TX': np.float, 'TN': np.float}
        )

        weather_hourly_df.rename(columns={'DATETIME': 'DATE'}, inplace=True)
        weather_hourly_df['DATE'] = weather_hourly_df['DATE'].astype('datetime64[D]')
        weather_daily_df['DATE'] = weather_daily_df['DATE'].astype('datetime64[D]')
        weather_daily_df.set_index(['DATE'], inplace=True)

        weather_df = weather_daily_df.merge(weather_hourly_df, on='DATE')
        weather_df.sort_values(['DATE', 'HOUR'], inplace=True)
        weather_df.set_index(['DATE', 'HOUR'], inplace=True)

        self.weather_df = weather_df
        self.weather_daily_df = weather_daily_df
        self.weather_hourly_df = weather_hourly_df

    @xs.runtime(args=('step', 'step_start', 'step_delta'))
    def run_step(self, step, step_start, step_delta):

        self.T_air = self.weather_df['T'][step_start].to_numpy()
        self.TM_air = self.weather_df['TM'][step_start].to_numpy()
        self.GR = self.weather_df['GR'][step_start].to_numpy()
        self.RH = self.weather_df['RH'][step_start].to_numpy()
        self.T_fruit = self.T_air
        self.TM = (self.weather_daily_df['TX'][step_start] + self.weather_daily_df['TN'][step_start]) / 2
