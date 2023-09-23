import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

###############################################################################
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
###############################################################################
print(f'Parsing: data\\5-filtered_ticker_list.json\ndata\\5-cleaned_stock_data.json\n')
ticker_sector = pd.read_parquet(Path.cwd().parent / 'data' / 'symbols.parquet')
ticker_sector = ticker_sector[['symbol', 'gics_sector']].set_index('symbol').to_dict()['gics_sector']

stock_dat = pd.read_parquet(Path.cwd().parent / 'data' / 'ohlc.parquet')
###############################################################################

def parse_ohlc_group(group):
    group = group.reset_index()
    group['date'] = pd.to_datetime(group['date'])
    group['adj_close'] = np.log(group['adjusted_close'])
    group.drop('symbol', axis='columns', inplace=True)

    # TODO: wrong way to resample OHLC data?
    group = group.groupby(pd.Grouper(key = 'date', freq = 'W-FRI')).mean().reset_index() 

    group["return_t"] = group['adj_close'].diff()
    group["return_t_plus_1"] = group['adj_close'].diff().shift(-1)
    group['sector'] = ticker_sector[ticker]
    return group[['return_t','return_t_plus_1','date','sector']].set_index(['date'])

_stock_ = stock_dat.groupby('symbol').apply(parse_ohlc_group)
_stock_.reset_index(inplace=True)
_stock_.rename({'symbol': 'ticker'}, axis=1, inplace=True)

_stock_ = _stock_[(_stock_['date'].dt.year<2020) & (_stock_['date'].dt.year>=2000)]
_stock_.dropna(inplace=True)

###############################################################################

print(f'Parsing: data\\6-technical_indicators.json\n')
technical_indicators = pd.read_parquet(Path.cwd().parent / 'data' / 'technical_indicators.parquet')

def parse_technical_indicator_group(group: pd.DataFrame):
    group = group.reset_index(inplace=False)
    group['date'] = pd.to_datetime(group['date'])
    group.drop('symbol', inplace=True, axis=1)
    return group.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).median() # TODO: shouldn't it be close?

_technical_indicators_ = technical_indicators.groupby(level='symbol').apply(parse_technical_indicator_group)
_technical_indicators_.reset_index(inplace=True)
_technical_indicators_.rename({'symbol': 'ticker'}, axis=1, inplace=True)

###############################################################################

# TODO: missing - need to pull from bloomberg

print(f'Parsing: data\\7-fundamental_indices_data.json\n')
_fuundamental_indices_ = pd.read_parquet(Path.cwd().parent / 'data' / 'fundamental_indices_data.parquet')
_fuundamental_indices_['date'] = pd.to_datetime(_fuundamental_indices_['date'])
_fuundamental_indices_ = _fuundamental_indices_[ _fuundamental_indices_['date'] > datetime(1999, 12, 1, 0, 0) ]
_fuundamental_indices_['date'] = _fuundamental_indices_['date'].astype(str)

###############################################################################
''' sentiment data by ticker'''

# TODO: missing

print(f'Parsing: data\\14-cleaned_all_500_company_news_sentiment.json\n')
sentiment_data =  json.loads(open('data\\14-cleaned_all_500_company_news_sentiment.json').read())
sentiment = pd.DataFrame(sentiment_data)

sentiment['date'] = pd.to_datetime(sentiment['pub_date'])
sentiment = sentiment.sort_values('date')     
sentiment['sentiment'] = [item[0] - item[1] for item in list(sentiment['logit'])]

sentiment = sentiment[['ticker','date', 'sentiment']].groupby(['ticker',pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()
sentiment = sentiment[pd.DatetimeIndex(sentiment['date']).year<2020]
sentiment = sentiment[pd.DatetimeIndex(sentiment['date']).year>=2000]
sentiment['date'] = sentiment['date'].dt.date.astype(str)

###############################################################################

_dat_ = pd.merge(_stock_,_technical_indicators_,how = "left",on=['ticker','date'])
_dat_ = pd.merge(_dat_,_fuundamental_indices_,how = "left",on=['ticker','date'])
_dat_ = pd.merge(_dat_,sentiment,how = "left",on=['ticker','date'])

_dat_['sector'] = [ticker_sector[ticker]['sector'] for ticker in _dat_['ticker']]
_dat_['industry'] = [ticker_sector[ticker]['industry'] for ticker in _dat_['ticker']]

print(f'Saving: data\\14-cleaned_all_500_company_news_sentiment.json\n')
_dat_.to_csv("dat_518_companies.csv")

