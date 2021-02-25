import pandas as pd
from pathlib import Path


class ProbabilityTableBase():

    _factors = {
        'Burst_Month': 'appearance_month',
        'Position_A': 'position',
        'Position_Ancestor_A': 'ancestor_is_apical',
        'Nature_Ancestor_F': 'ancestor_nature'
    }

    def get_factor_values(self, tbl, gu):
        index_lables = tbl.index.names if tbl.index.names else [tbl.index.name]
        factors = [factor for factor in index_lables if factor in self._factors.values()]
        return tuple([getattr(self, factor)[gu] for factor in factors])

    def get_probability_tables(self, path):
        dfs = {}
        path = Path(path)
        for file in path.iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file).rename(self._factors, axis=1)
                if 'number' in df.columns:
                    df.drop(columns='number', inplace=True)
                tbl = pd.DataFrame(columns=df.columns)
                if 'appearance_month' in df:  # split compound 'x-y-z' month into rows
                    for i, d in df['appearance_month'].items():
                        for month in list(map(int, d.split('-'))):
                            row = df.loc[i].copy()
                            row['appearance_month'] = month
                            tbl = tbl.append(row, ignore_index=True)
                else:
                    tbl = df
                cycle = int(file.name.split('_0')[1][0])
                index = [factor for factor in self._factors.values() if factor in tbl.columns]
                if len(index):
                    tbl.set_index([factor for factor in self._factors.values() if factor in tbl.columns], inplace=True)
                dfs[cycle] = tbl.astype(float)
        return dfs
