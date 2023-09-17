import os
from pathlib import Path
import json
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ndiffs
import numpy as np
from tqdm import tqdm

def load_data_and_configs(data_path, json_path):
    with open(json_path, 'r') as f:
        ticker_sector = json.load(f)
    data = pd.read_csv(data_path, index_col=0)
    data['date'] = pd.to_datetime(data['date'])
    return data, ticker_sector

def fit_and_predict_arima(train_data, test_data):
    try:
        _return_dat = np.concatenate([train_data['return_t'].iloc[0], train_data['return_t_plus_1'].values])
    except:
        _return_dat = train_data['return_t'].values
    
    kpss_diffs = ndiffs(_return_dat, alpha=0.05, test='kpss', max_d=4)
    adf_diffs = ndiffs(_return_dat, alpha=0.05, test='adf', max_d=4)
    n_diffs = max(adf_diffs, kpss_diffs)
    
    model = pm.auto_arima(_return_dat, start_p=1, start_q=1, d=n_diffs, seasonal=False, stepwise=True,
                          suppress_warnings=True, error_action="ignore", max_p=4, information_criterion="aic", 
                          max_q=4, trace=False)
    
    _test_y = test_data['return_t_plus_1'].values
    _pred_y = [model.predict(1)[0] for _ in _test_y]
    for pos in _test_y:
        model.update(pos)
    
    return np.asarray(_pred_y)


project_dir = Path.cwd().parent
data_path = project_dir / "2-cleaned_data" / "dat_518_companies.csv"
json_path = project_dir / "2-cleaned_data" / "ticker_sector_information.json"
_dat_, ticker_sector = load_data_and_configs(data_path, json_path)

available_years = [item for item in range(2003,2020)]
_output = pd.DataFrame()

for _year in tqdm(available_years):
    train_df = _dat_[_dat_['date'].dt.year < _year].copy()
    test_df = _dat_[_dat_['date'].dt.year == _year].copy()

    for ticker in tqdm(ticker_sector, leave=False):
        temp = train_df[train_df['ticker'] == ticker]
        temp_test = test_df[test_df['ticker'] == ticker]

        if len(temp) > 20:
            temp = temp.sort_values('date') 
            temp_test = temp_test.sort_values('date') 
            
            _pred_y = fit_and_predict_arima(temp, temp_test)
            
            temp_test['ARIMA_pred'] = _pred_y
            _output = pd.concat([_output,temp_test], axis=0)
            
            print(np.mean((_output['ARIMA_pred']>=0) == (_output['return_t_plus_1']>=0)))
            
###############################################################################
_output_2 = _output[_output['date'].dt.year >= 2003]

print('ARIMA yearly update')
print('MAE:', np.mean(np.abs(_output_2['ARIMA_pred'] - _output_2['return_t_plus_1'])))
print('RMSE:', np.sqrt(np.mean(np.square(_output_2['ARIMA_pred'] - _output_2['return_t_plus_1']))))
print('DA:', np.mean((_output_2['ARIMA_pred'] >= 0) == (_output_2['return_t_plus_1'] >= 0)))
print('UDA:', sum( np.logical_and((_output_2['ARIMA_pred'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA:', sum( np.logical_and((_output_2['ARIMA_pred'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))


_output = _output[['date','ticker','ARIMA_pred']].copy()
_output['date'] = _output['date'].astype(str)
_output.to_csv(project_dir / "saved_output" / "ARIMA.csv")
