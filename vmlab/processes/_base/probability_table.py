import pandas as pd
import numpy as np
from pathlib import Path
import xsimlab as xs

from .parameter import ParameterizedProcess
from vmlab.enums import Position, Nature


@xs.process
class BaseProbabilityTableProcess():

    rng = None

    table_dir_path = xs.variable(intent='in', static=True, default=None)

    # new factor names (values in dict) must match variable names in process
    _factors = {
        'Burst_Month': 'appearance_month',
        'Position_A': 'position',
        'Position_Ancestor_A': 'ancestor_is_apical',
        'Nature_Ancestor_F': 'ancestor_nature',
        'Flowering_Week': 'flowering_week',
        'Nature_F': 'nature',
        'Nb_Inflorescences': 'nb_inflorescences',
        'Has_Apical_GU_Child': 'has_apical_child_between'
    }
    # factor values
    _factor_values = {
        'appearance_month': range(1, 13),
        'position': list(Position.values()),
        'ancestor_is_apical': [0., 1.],
        'ancestor_nature': list(Nature.values()),
        'flowering_week': range(0, 13),  # 0 = no flowering
        'nature': list(Nature.values()),
        'nb_inflorescences': range(0, 10),
        'has_apical_child_between': [0., 1.]
    }

    def get_indices(self, tbl, gu_idx_list):
        """Build a list of indices from factor (process variables) values to query the panda probability table"""
        if len(tbl.index) <= 1 and tbl.index.name is None:  # no factors at all, just one row with THE probability
            return np.zeros(len(gu_idx_list))
        index_lables = tbl.index.names if tbl.index.names else [tbl.index.name]
        indices = np.column_stack(tuple([getattr(self, factor)[gu_idx_list] for factor in index_lables]))
        return indices if len(index_lables) > 1 else indices.flatten()

    def get_probability_tables(self):
        tbls = {}
        path = Path(self.table_dir_path)
        for file in path.iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file).rename(self._factors, axis=1)

                # drop the number column
                if 'number' in df.columns:
                    df.drop(columns='number', inplace=True)

                tbl = pd.DataFrame(columns=df.columns)

                # split compound 'x-y-z' values into rows
                if 'appearance_month' in df:
                    for i, d in df['appearance_month'].items():
                        for month in list(map(np.float, d.split('-'))):
                            row = df.loc[i].copy()
                            row['appearance_month'] = month
                            tbl = tbl.append(row, ignore_index=True)
                elif 'flowering_week' in df:
                    for i, d in df['flowering_week'].items():
                        for week in list(map(np.float, d.split('-'))):
                            row = df.loc[i].copy()
                            row['flowering_week'] = week
                            tbl = tbl.append(row, ignore_index=True)
                else:
                    tbl = df

                # extract cycle from file name
                cycle = float(file.name.split('_0')[1][0])

                # create MultiIndex from factors in table
                index = [factor for factor in self._factors.values() if factor in tbl.columns]
                if len(index):
                    tbl.set_index([factor for factor in self._factors.values() if factor in tbl.columns], inplace=True, )

                # cast all values to float
                tbl.astype(np.float, copy=False)

                # normalize probabilities (0 <= sum(p) >= 1) for nominal and multinominal distributions (lambda is possion)
                if 'lambda' not in tbl.columns:
                    tbl.where(tbl.sum(axis=1) <= 1., tbl.divide(tbl.sum(axis=1), axis='rows'), inplace=True)

                # extend tbl so all possible factor combinations are present
                if len(index) > 1:
                    tbl = tbl.join(
                        pd.DataFrame(index=tbl.index.from_product([self._factor_values[name] for name in tbl.index.names])),
                        on=tbl.index.names, how='outer', sort=tbl.index.names
                    )
                elif len(index) == 1:
                    tbl = tbl.reindex(pd.Index(self._factor_values[index[0]], name=index[0]), fill_value=0.)

                tbl[tbl.isna()] = 0.
                tbls[cycle] = tbl

        return tbls


@xs.process
class ProbabilityTableProcess(ParameterizedProcess):

    # new factor names (values in dict) must match variable names in process
    _factors = {
        'Burst_Month': 'appearance_month',
        'Position_A': 'is_apical',
        'Position_Ancestor_A': 'ancestor_is_apical',
        'Nature_Ancestor_F': 'ancestor_nature',
        'Flowering_Week': 'flowering_week',
        'Nature_F': 'nature',
        'Nb_Inflorescences': 'nb_inflorescences',
        'Has_Apical_GU_Child': 'has_apical_child_between'
    }
    # factor values
    _factor_values = {
        'appearance_month': range(1, 13),
        'is_apical': [0., 1.],
        'ancestor_is_apical': [0., 1.],
        'ancestor_nature': list(Nature.values()),
        'flowering_week': range(0, 13),  # 0 = no flowering
        'nature': list(Nature.values()),
        'nb_inflorescences': range(0, 10),
        'has_apical_child_between': [0., 1.]
    }

    def get_indices(self, tbl, gu_idx_list):
        """Build a list of indices from factor (process variables) values to query the panda probability table"""
        if len(tbl.index) <= 1 and tbl.index.name is None:  # no factors at all, just one row with THE probability
            return np.zeros(len(gu_idx_list))
        index_lables = tbl.index.names if tbl.index.names else [tbl.index.name]
        indices = np.column_stack(tuple([getattr(self, factor)[gu_idx_list] for factor in index_lables]))
        return indices if len(index_lables) > 1 else indices.flatten()

    def get_binomial(self, tbl, gu_indices):
        indices = self.get_indices(tbl, gu_indices)
        probability = tbl.loc[indices.tolist()].values.flatten()
        return self.rng.binomial(1, probability, probability.shape)

    def get_multinomial(self, tbl, gu_index):
        index = self.get_indices(tbl, np.array(gu_index))
        probabilities = tbl.loc[index.tolist()].values.flatten()
        return self.rng.multinomial(1, probabilities) if probabilities.sum() > 0 else np.zeros(probabilities.shape)

    def get_poisson(self, tbl, gu_indices):
        indices = self.get_indices(tbl, gu_indices)
        lam = tbl.loc[indices.tolist()].values.flatten()
        return np.round(np.where(lam == 0., 0., self.rng.poisson(lam, lam.shape) + 1.), 0)

    def get_probability_tables(self):

        dir_path = Path(self.parameter_file_path).parent
        tbls = {}
        for var_name, tbl_dir_path in self.parameters.probability_tables.items():
            tbls[var_name] = {}
            path = dir_path.joinpath(tbl_dir_path)
            for file in path.iterdir():
                if file.suffix == '.csv':
                    df = pd.read_csv(file).rename(self._factors, axis=1)

                    # drop the number column
                    if 'number' in df.columns:
                        df.drop(columns='number', inplace=True)

                    tbl = pd.DataFrame(columns=df.columns)

                    # split compound 'x-y-z' values into rows
                    if 'appearance_month' in df:
                        for i, d in df['appearance_month'].items():
                            for month in list(map(np.float, d.split('-'))):
                                row = df.loc[i].copy()
                                row['appearance_month'] = month
                                tbl = tbl.append(row, ignore_index=True)
                    elif 'flowering_week' in df:
                        for i, d in df['flowering_week'].items():
                            for week in list(map(np.float, d.split('-'))):
                                row = df.loc[i].copy()
                                row['flowering_week'] = week
                                tbl = tbl.append(row, ignore_index=True)
                    else:
                        tbl = df

                    # extract cycle from file name
                    cycle = float(file.name.split('_0')[1][0])

                    # create MultiIndex from factors in table
                    index = [factor for factor in self._factors.values() if factor in tbl.columns]
                    if len(index):
                        tbl.set_index([factor for factor in self._factors.values() if factor in tbl.columns], inplace=True, )

                    # cast all values to float
                    tbl.astype(np.float, copy=False)

                    # normalize probabilities (0 <= sum(p) >= 1) for nomial and multinomial distributions (lambda is possion)
                    if 'lambda' not in tbl.columns:
                        tbl.where(tbl.sum(axis=1) <= 1., tbl.divide(tbl.sum(axis=1), axis='rows'), inplace=True)

                    # extend tbl so all possible factor combinations are present
                    if len(index) > 1:
                        tbl = tbl.join(
                            pd.DataFrame(index=tbl.index.from_product([self._factor_values[name] for name in tbl.index.names])),
                            on=tbl.index.names, how='outer', sort=tbl.index.names
                        )
                    elif len(index) == 1:
                        tbl = tbl.reindex(pd.Index(self._factor_values[index[0]], name=index[0]), fill_value=0.)

                    tbl[tbl.isna()] = 0.
                    tbls[var_name][cycle] = tbl

        return tbls

    def initialize(self):
        self.rng = np.random.default_rng(seed=self.seed)
        super(ProbabilityTableProcess, self).initialize()
