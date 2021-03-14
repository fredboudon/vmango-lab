import pandas as pd
import numpy as np
from pathlib import Path
import xsimlab as xs

from vmlab.enums import Position, Nature


@xs.process
class BaseProbabilityTableProcess():

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
        'flowering_week': range(1, 13),
        'nature': list(Nature.values()),
        'nb_inflorescences': [0., 1., 2., 3.],
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

                # normalize probabilities (0 <= sum(p) >= 1)
                if 'probability' in tbl.columns:  # binominal
                    tbl[tbl > 1.] = 1.
                else:  # multinominal
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
