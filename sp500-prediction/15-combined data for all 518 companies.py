from pathlib import Path
import pandas as pd
import numpy as np

###############################################################################
returns: pd.DataFrame = pd.read_parquet(Path.cwd() / 'data' / 'ohlc.parquet')[['adjusted_close']]
technicals: pd.DataFrame = pd.read_parquet(Path.cwd() / 'data' / 'technical_indicators.parquet')
fundamentals: pd.DataFrame = pd.read_parquet(Path.cwd() / 'data' / 'fundamentals.parquet')
sentiments: pd.DataFrame = pd.read_parquet(Path.cwd() / 'data' / 'sentiment.parquet')
sectors: pd.DataFrame = pd.read_parquet(Path.cwd() / 'data' / 'symbols.parquet')[['symbol', 'gics_sector']]
macros: pd.DataFrame = pd.read_parquet(Path.cwd() / 'data' / 'macro.parquet')

sentiments.index.names = ['symbol', 'date']

###############################################################################
# resample daily to weekly

def resample_weekly(df: pd.DataFrame):
    assert df.index.names == ['symbol', 'date']
    df_reset = df.reset_index()
    return df_reset.groupby('symbol').resample('W-FRI', on='date').median()

returns_weekly = resample_weekly(returns)
technicals_weekly = resample_weekly(technicals)
fundamentals_weekly = resample_weekly(fundamentals)
sentiments_weekly = resample_weekly(sentiments)
macros_weekly = macros.resample('W-FRI').median().reset_index()

###############################################################################

_dat_: pd.DataFrame = returns_weekly.merge(technicals_weekly, how='left', left_index=True, right_index=True)
_dat_ = _dat_.merge(fundamentals_weekly, how='left', left_index=True, right_index=True)
_dat_ = _dat_.merge(sentiments_weekly, how='left', left_index=True, right_index=True)
_dat_ = _dat_.reset_index().merge(sectors, how='left', on='symbol').set_index(['symbol','date'])
_dat_ = _dat_.reset_index().merge(macros_weekly, how='left', on='date').set_index(['symbol', 'date'])

###############################################################################
# calculate returns
temp = _dat_['adjusted_close'].groupby(level='symbol').apply(lambda group: np.log(group) - np.log(group).shift())
temp.index = temp.index.droplevel(1)
_dat_['return_t'] = temp
temp = _dat_['adjusted_close'].groupby(level='symbol').apply(lambda group: np.log(group).shift(-1) - np.log(group))
temp.index = temp.index.droplevel(1)
_dat_['return_t_plus_1'] = temp
_dat_ = _dat_[~(_dat_[['return_t', 'return_t_plus_1']].isna().any(axis=1))].copy()

_dat_ = _dat_.drop(columns=['adjusted_close'],)

###############################################################################
# filter dates
_dat_ = _dat_.reset_index()
_dat_ = _dat_[(_dat_.date.dt.year >= 2000) & (_dat_.date.dt.year < 2020)]

###############################################################################
# fix names to be consistent downstream
mappings = {'PX_TO_BOOK_RATIO': 'PbRatio', 
            'PE_RATIO': 'PeRatio', 
            'PX_TO_SALES_RATIO': 'PsRatio', 
            'sentiment_score': 'sentiment',
            'gics_sector': 'sector',
            'symbol': 'ticker'}

_dat_.columns = _dat_.columns.map(lambda key: mappings.get(key, key))

_dat_['sector'] = _dat_['sector'].astype(str)

print(f'Saving: data\\dat_518_companies.parquet\n')
_dat_.to_parquet("data\\dat_518_companies.parquet")