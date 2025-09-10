# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import math
import numpy as np
import pandas as pd
from typing import Dict, List
from functools import reduce
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt

from tvDatafeed import TvDatafeed, Interval
    
def download_btc_cap(nb = 1200):
    tv=TvDatafeed()
    btc_cap = tv.get_hist("BTC.D", "CRYPTOCAP", interval=Interval.in_daily,n_bars = nb)
    
    btc_cap['date'] = btc_cap.index
    btc_cap['timeindex'] = (btc_cap['date'].values.astype(float)/10**6 - 1279670400000) // 86400000
    btc_cap['btc_cap'] = (btc_cap['open'] + btc_cap['high'] + btc_cap['low'] + btc_cap['close'])/4
    btc_cap_export = btc_cap[['btc_cap', 'timeindex']]

    return btc_cap_export
    
class TVdatafeed_exemple(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {}

    sell_params = {}

    protection_params = {}
    
    minimal_roi = {
        "0": 20
    }

    stoploss = -0.14
    
    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.298
    trailing_stop_positive_offset = 0.349
    trailing_only_offset_is_reached = False
    
    timeframe = '1d'
    
    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    plot_config = {
        'main_plot':{},
        'subplots':{
            "btc capitalisation": {
                'btc_cap':{'color': 'blue'},
                },
        }
    }
    
    init_trailing_dict = {
    'trailing_buy_order_started': False,
    'trailing_buy_order_uplimit': 0,
    'start_trailing_price': 0,
    'buy_tag': None,
    }



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['timeindex'] = (dataframe['date'].values.astype(float)/10**6 - 1279670400000) // 86400000

        if self.config['runmode'].value in ('live', 'dry_run', 'backtest'):

            if "btc_cap" not in dataframe.index :
            # We haven't uploaded TVdatafeed yet
                btc_cap_export = download_btc_cap()
                dataframe = pd.merge(dataframe, btc_cap_export, on='timeindex')

            else : 
            # ALready imported but in live/dry run mode we have to check data are up to date
                if self.config['runmode'].value in ('live', 'dry_run'):
                    last_candle = dataframe.iloc[-1].squeeze()
                    
                    if math.isnan(last_candle['btc_cap']):
                        btc_cap_export = download_btc_cap()
                        dataframe = pd.merge(dataframe, btc_cap_export, on='timeindex')
        
        else :
            #Download will be performed only once in backtest / hyperopt
            btc_cap_export = download_btc_cap()
            dataframe = pd.merge(dataframe, btc_cap_export, on='timeindex')
                
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
            
        dataframe.loc[:, 'buy_tag'] = ''
        
        buy_conditions = ()
        dataframe.loc[buy_conditions, 'buy_tag'] += 'Buy low'
        conditions.append(buy_conditions)
        
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                ['buy']
            ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        
        sell_condition = ()
        dataframe.loc[sell_condition, 'exit_tag'] += 'Sell_high'
        conditions.append(sell_condition)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                ['sell']
            ] = 1
        return dataframe