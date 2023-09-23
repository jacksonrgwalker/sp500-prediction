import  os
from pathlib import Path

BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "saved_output"
FIGURE_DIR = BASE_DIR / 'figures'
###############################################################################
from sklearn.linear_model import LinearRegression

from tqdm import tqdm
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

###############################################################################
_output = pd.read_parquet(DATA_DIR / "dat_518_companies.parquet")
_tickers = _output.ticker.unique()
###############################################################################

_output = _output[['date','ticker','return_t_plus_1']].copy()
_ARIMA_output = pd.read_parquet(OUTPUT_DIR / "ARIMA.parquet")
_LR_output = pd.read_parquet(OUTPUT_DIR / "LR_pred.parquet")
_RF_output = pd.read_parquet(OUTPUT_DIR / "RF_pred.parquet")
_Dense_output = pd.read_parquet(OUTPUT_DIR / "Dense_monthly_update_10_year.parquet")
_LSTM_two_layer = pd.read_parquet(OUTPUT_DIR / "LSTM_pred_two_layer_mae_linear_activation_length_4.parquet")
_LSTM_stock_output = pd.read_parquet(OUTPUT_DIR / "LSTM_pred_one_layer_mae_linear_activation_length_3_stock.parquet")
###############################################################################

_output = pd.merge(_output,_ARIMA_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_LR_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_RF_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_Dense_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_LSTM_two_layer,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_LSTM_stock_output,how = "left",on=['date','ticker'])
###############################################################################

available_years = [item for item in range(2003,2020)]

selected_model = ['RF_pred','Dense_monthly_update_10_year','LSTM_two_layer_length_3_pred','LSTM_pred_stock']

_output_2 = pd.DataFrame()
coefficient = []
for _year in tqdm(available_years):
    temp = _output[pd.DatetimeIndex(_output['date']).year < _year ].copy()
    temp = temp[pd.DatetimeIndex(temp['date']).year >= (_year-2) ].copy()
    _train_X = np.asarray(temp[selected_model])
    _train_X  = np.nan_to_num(_train_X)
    _train_y = np.asarray(temp['return_t_plus_1']).reshape(-1,)
    test_df = _output[ pd.DatetimeIndex(_output['date']).year == _year].copy()  
    _test_X = np.asarray(test_df[selected_model])
    _test_X  = np.nan_to_num(_test_X)
    _test_y = np.asarray(test_df['return_t_plus_1']).reshape(-1,)
    
    model = LinearRegression(fit_intercept = False, positive = True)      
    model.fit(_train_X,_train_y)
    
    coefficient.append( list(model.coef_) )
    _pred_y = model.predict(_test_X)
    test_df['ensemble'] = _pred_y
    _output_2 = pd.concat([_output_2,test_df], axis=0)

print('One layer LSTM model fine tune on each individual stock')
print('MAE')
print( np.mean( np.abs(_output_2['ensemble'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['ensemble'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['ensemble']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['ensemble'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['ensemble'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))

###############################################################################
coefficient = pd.DataFrame(coefficient)
coefficient.columns = ['RF each sector','FFNN','LSTM','LSTM fine tuning on each stock']
coefficient.index = available_years

plt.plot(coefficient)
plt.legend(coefficient.columns)
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.ylim([0,1])
plt.savefig('model_weight.png', dpi=400,bbox_inches='tight') 
    
###############################################################################
'''plot for the yearly DA performance for each individual model'''
available_years = [item for item in range(2003,2020)]

ARIMA = []
LR = []
RF = []
FFNN= []
LSTM = []
for _year in tqdm(available_years):
    temp = _output[pd.DatetimeIndex(_output['date']).year == _year ].copy()
    ARIMA.append(np.mean((temp['ARIMA_pred']>=0) == (temp['return_t_plus_1']>=0)))
    LR.append(np.mean((temp['LR_pred']>=0) == (temp['return_t_plus_1']>=0)))
    RF.append(np.mean((temp['RF_pred']>=0) == (temp['return_t_plus_1']>=0)))
    FFNN.append(np.mean((temp['Dense_monthly_update_10_year']>=0) == (temp['return_t_plus_1']>=0)))
    LSTM.append(np.mean((temp['LSTM_two_layer_length_3_pred']>=0) == (temp['return_t_plus_1']>=0)))
    
fig = plt.figure()
ax = fig.gca()
ax.plot(available_years,ARIMA,label="ARIMA")
ax.plot(available_years,LR,label="Linear")
ax.plot(available_years,RF,label="Random Forest")
ax.plot(available_years,FFNN,label="FFNN")
ax.plot(available_years,LSTM,label="LSTM")
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(FIGURE_DIR / 'model comparison.png', dpi=400,bbox_inches='tight') 

###############################################################################
'''yearly DA for ensemble model and the best individual model'''
available_years = [item for item in range(2003,2020)]

ensemble = []
best_individual = []
for _year in tqdm(available_years):
    temp = _output_2[pd.DatetimeIndex(_output_2['date']).year == _year ].copy()
    RF = (np.mean((temp['RF_pred']>=0) == (temp['return_t_plus_1']>=0)))
    Dense= (np.mean((temp['Dense_monthly_update_10_year']>=0) == (temp['return_t_plus_1']>=0)))
    LSTM_stock = (np.mean((temp['LSTM_pred_stock']>=0) == (temp['return_t_plus_1']>=0)))
    LSTM = (np.mean((temp['LSTM_two_layer_length_3_pred']>=0) == (temp['return_t_plus_1']>=0)))
    best_individual.append(np.max([RF,Dense,LSTM_stock,LSTM]))
    ensemble.append(np.mean((temp['ensemble']>=0) == (temp['return_t_plus_1']>=0)))

###############################################################################
'''plot for the ensemble model'''
for ticker in tqdm(_tickers):
    temp = _output_2[_output_2['ticker']==ticker].copy()
    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp.sort_values('date')    
    _year = temp.iloc[0]['date'].year
    if len(temp[pd.DatetimeIndex(temp['date']).year == _year ]) <= 20:
        temp = temp[pd.DatetimeIndex(temp['date']).year > _year ].copy()
    temp['direction'] = (temp['ensemble']>=0) == (temp['return_t_plus_1']>=0)
    temp = temp[['date','direction']].groupby([pd.Grouper(key = 'date', freq = 'YS')]).mean().reset_index()
    plt.plot(pd.DatetimeIndex(temp['date']).year,temp['direction'],alpha= 0.2, linewidth=0.2)
    
plt.grid(color='gray', linestyle='-.', linewidth=0.2)
plt.plot(available_years,ensemble,color="red",label="Ensemble model")
plt.plot(available_years,best_individual,color="blue",label="Best individual model")
plt.legend()
plt.ylim([0.425,0.775])
plt.savefig(FIGURE_DIR / 'ensemble_model_performance.png', dpi=400,bbox_inches='tight') 

###############################################################################
'''plot for where the ensemble model predicted a over 5 percent return'''
output_3 = _output_2[_output_2['ensemble'] >= 0.05].copy()

count = 0
for ticker in _tickers:
    output_4 = output_3[output_3['ticker']==ticker].copy()
    if len(output_4)>0:
        plt.scatter(pd.DatetimeIndex(output_4['date']),output_4['return_t_plus_1'],s= 0.2,alpha=0.8)
        count = count +1
plt.ylim([-0.5,0.5])
plt.axhline(np.mean(output_3['return_t_plus_1']), color='r', linestyle='-.', linewidth= 0.5)
plt.ylabel("Log Return")
plt.savefig(FIGURE_DIR / 'ensemble_model_five_percent.png', dpi=400,bbox_inches='tight') 



