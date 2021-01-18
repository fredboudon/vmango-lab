import xsimlab as xs
import pandas as pd
import numpy as np
import pathlib
import datetime

from . import parameters
from . import architecture


@xs.process
class Environment():

    hour = xs.index(dims=('hour'))
    GU = xs.foreign(architecture.Architecture, 'GU')
    nb_fruits_ini = xs.foreign(architecture.Architecture, 'nb_fruits_ini')
    nb_leaves = xs.foreign(architecture.Architecture, 'nb_leaves')
    params = xs.foreign(parameters.Parameters, 'environment')

    weather_df = None
    weather_daily_df = None
    weather_hourly_df = None

    nb_fruits = xs.variable(
        dims=('GU'),
        intent='out'
    )

    bloom_date = xs.variable(
        dims=('GU'),
        intent='inout',
        description='bloom date',
        attrs={
            'unit': 'date'
        },
        static=True
    )

    DAB = xs.variable(
        dims=('GU'),
        intent='out',
        description='days after bloom',
        attrs={
            'unit': 'd'
        }
    )

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

    dd_cum = xs.variable(
        dims=('GU'),
        intent='out',
        description='cumulated degree-days of the current day',
        attrs={
            'unit': 'dd'
        }
    )

    dd_delta = xs.variable(
        dims=('GU'),
        intent='out',
        description='daily variation in degree days',
        attrs={
            'unit': 'dd day-1'
        }
    )

    DAB = xs.variable(
        dims=('GU'),
        intent='out',
        description='Days after bloom in current cycle',
        attrs={
            'unit': 'm²'
        }
    )

    LFratio = xs.variable(
        dims=('GU'),
        intent='out'
    )

    LA = xs.variable(
        dims=('GU'),
        intent='out',
        description='total leaf area per branch',
        attrs={
            'unit': 'm²'
        }
    )

    def initialize(self):

        _, params = self.params

        self.nb_fruits = np.zeros(self.GU.shape)
        self.dd_delta = np.zeros(self.GU.shape)
        self.dd_cum = np.zeros(self.GU.shape)
        self.LFratio = np.array([nb_leaves / nb_fruits if nb_fruits > 0 else 0.
                                 for nb_leaves, nb_fruits in zip(self.nb_leaves, self.nb_fruits)])

        self.hour = np.arange(0, 24, 1, dtype=np.int8)
        self.bloom_date = np.array([np.datetime64(datetime.date.fromisoformat(bloom_date)).astype('datetime64[D]')
                                    for bloom_date in self.bloom_date])

        weather_hourly_df = pd.read_csv(
            pathlib.Path(self.params[0].parent).joinpath(self.params[1].weather_hourly_file_path),
            sep=';',
            parse_dates=['DATETIME'],
            dayfirst=True,
            usecols=['HOUR', 'GR', 'T', 'RH', 'DATETIME'],
            dtype={'GR': np.float, 'T': np.float, 'RH': np.float}
        )
        weather_daily_df = pd.read_csv(
            pathlib.Path(self.params[0].parent).joinpath(self.params[1].weather_daily_file_path),
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

        # weather_hour_count = weather.groupby(['DATE']).count()
        # if len(weather_hour_count[weather_hour_count['HOUR'] != 24].values) > 0:
        #     print('Input data has days with less than 24 h')

        self.weather_df = weather_df
        self.weather_daily_df = weather_daily_df
        self.weather_hourly_df = weather_hourly_df

    @xs.runtime(args=('step', 'step_start', 'step_end', 'step_delta'))
    def run_step(self, step, step_start, step_end, step_delta):

        _, params = self.params
        dd_cum_0 = params.dd_cum_0
        Tbase_fruit = params.Tbase_fruit

        self.T_air = self.weather_df['T'][step_start].to_numpy()
        self.TM_air = self.weather_df['TM'][step_start].to_numpy()
        self.GR = self.weather_df['GR'][step_start].to_numpy()
        self.RH = self.weather_df['RH'][step_start].to_numpy()
        self.T_fruit = self.T_air

        TM = (self.weather_daily_df['TX'][step_start] + self.weather_daily_df['TN'][step_start]) / 2
        self.dd_delta = np.where(
            step_start >= self.bloom_date,
            max(0, TM - Tbase_fruit),
            0.
        )
        self.dd_cum = np.where(
            step_start >= self.bloom_date,
            self.dd_cum + self.dd_delta,
            0.
        )
        self.nb_fruits = np.where(
            self.dd_cum >= dd_cum_0,
            self.nb_fruits_ini,
            0.
        )
        self.LFratio = np.array([nb_leaves / nb_fruits if nb_fruits > 0 else 0.
                                 for nb_leaves, nb_fruits in zip(self.nb_leaves, self.nb_fruits)])

        # leaf area (eq. 11) :
        self.LA = self.params[1].e_nleaf2LA_1 * self.LFratio ** self.params[1].e_nleaf2LA_2

        # print(step_start, step_start >= self.bloom_date, self.LFratio)
