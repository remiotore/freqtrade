from ctypes import Union
import numpy as np
from freqtrade.persistence.trade_model import Order
from freqtrade.strategy import IStrategy, merge_informative_pair ,informative
from typing import Dict, List, Optional
from functools import reduce
from pandas import DataFrame 
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy.parameters import IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import csv
import os
from freqtrade.configuration import Configuration
config = Configuration.from_files([])
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade




from datetime import datetime, timedelta

exchange = config['exchange']['name']




def save_dict_to_csv(data_dict, filename):
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_dict.keys())  # Write the header rows
            writer.writerow(data_dict.values())  # Write the data row
            
class AEMA(IStrategy):
    trailing_stop = False
    use_custom_stoploss = True
    process_only_new_candles = True
    ignore_roi_if_entry_signal = True
    custom_price_max_distance_ratio = 0.5
    use_exit_signal = True
    exit_profit_only = False
    INTERFACE_VERSION: int = 3
    EMA_SHORT_TERM = 10
    EMA_MEDIUM_TERM = 12
    EMA_LONG_TERM = 50
    VOLATILITY_THRESHOLD = 0.1  # 10% threshold
    startup_candle_count: int = 10
    minimal_roi = {"0": 100}
    order_types = {
    'entry': 'limit',
    'exit': 'limit',
    'stoploss': 'market',
    'stoploss_on_exchange': False
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }
    
    # stoploss = -0.99
    stoploss =  -0.333
    timeframe = '1h'
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False
    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    # Number of candles the strategy requires before producing valid signals
    

    
    def informative_pairs(self):

         # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        
        return informative_pairs
        
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf_tf = '1d'
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day rsi
        
        ticker = self.dp.ticker(metadata['pair'])
        
        # dataframe['close'] = ticker['last']
        # dataframe['close'] = ticker['last']
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        pairs_info = [(metadata['pair'], inf_tf)]
        informative['ema3_high'] = ta.EMA(informative['high'], timeperiod=3)
        informative['ema5_high'] = ta.EMA(informative['high'], timeperiod=5)
        informative['ema10_high'] = ta.EMA(informative['high'], timeperiod=10)
        informative['ema15_high'] = ta.EMA(informative['high'], timeperiod=15)
        informative['ema21_high'] = ta.EMA(informative['high'], timeperiod=21)
        informative['ema50_high'] = ta.EMA(informative['high'], timeperiod=50)
        informative['ema90_high'] = ta.EMA(informative['high'], timeperiod=90)
        informative['ema100_high'] = ta.EMA(informative['high'], timeperiod=100)
        # dataframe['ema200_high'] = ta.EMA(dataframe['high'], timeperiod=200)
        
        # # EMAS LOW 
        informative['ema3_low'] = ta.EMA(informative['low'], timeperiod=3)
        informative['ema5_low'] = ta.EMA(informative['low'], timeperiod=5)
        informative['ema10_low'] = ta.EMA(informative['low'], timeperiod=10)
        informative['ema15_low'] = ta.EMA(informative['low'], timeperiod=15)
        informative['ema21_low'] = ta.EMA(informative['low'], timeperiod=21)
        informative['ema50_low'] = ta.EMA(informative['low'], timeperiod=50)
        informative['ema90_low'] = ta.EMA(informative['low'], timeperiod=90)
        informative['ema100_low'] = ta.EMA(informative['low'], timeperiod=100)
        # dataframe['ema200_low'] = ta.EMA(dataframe['low'], timeperiod=200)
        
        informative['lower_band_short'] = informative['ema15_low'] * (1 - 0.1)
        informative['upper_band_short'] = informative['ema15_high'] * (1 + 0.1)
        informative['low_outlier_values']     = np.nan
        informative['high_outlier_values'] = np.nan
        informative['volatility_score'] = np.nan
        informative['volatility_scores'] = np.nan
        
        outliers = []
        current_outlier = None
        current_side = None
        cluster_count = 0
        current_cluster = 0
        volatility_score = 0
        low_outlier_values = []
        high_outlier_values = []
        
        # Iterate through the DataFrame row

        for i in range(len(informative)):
            low_price = informative['low'].iloc[i]
            high_price = informative['high'].iloc[i]
            lower_band_short = informative['lower_band_short'].iloc[i]
            upper_band_short = informative['upper_band_short'].iloc[i]
            
            if current_side is None:
                if low_price < lower_band_short:
                    current_side = 'below'
                    low_outlier_values.append(low_price)  # Append low price
                    informative.at[informative.index[i], 'low_outlier_values'] = low_price
                    current_cluster += 1  # Increase the cluster count
                    volatility_score += 1  # Increase the volatility score
                elif high_price > upper_band_short:
                    current_side = 'above'
                    high_outlier_values.append(high_price)  # Append high price
                    informative.at[informative.index[i], 'high_outlier_values'] = high_price
                    current_cluster += 1  # Increase the cluster count
                    volatility_score += 1  # Increase the volatility score
            elif current_side == 'below':
                if high_price > upper_band_short:
                    current_side = 'above'
                    high_outlier_values.append(high_price)  # Append high price
                    informative.at[informative.index[i], 'high_outlier_values'] = high_price
                    current_cluster += 1  # Increase the cluster count
                    volatility_score += 1  # Increase the volatility score
            elif current_side == 'above':
                if low_price < lower_band_short:
                    current_side = 'below'
                    low_outlier_values.append(low_price)  # Append low price
                    informative.at[informative.index[i], 'low_outlier_values'] = low_price
                    current_cluster += 1  # Increase the cluster count
                    volatility_score += 1  # Increase the volatility score

            # Assign the current cluster value to the dataframe
            informative.at[informative.index[i], 'volatility_score'] = current_cluster
            informative['volatility_scores'] = current_cluster
            volatility_info = {
                    'Pair': metadata['pair'],
                    'Volatility_Score': cluster_count,
                    
                    'TimeFrame': self.timeframe,
                }
            
            os.makedirs('ohlcv_age', exist_ok=True) 
            file_path = os.path.join('ohlcv_age', 'volatility_data.csv')
            save_dict_to_csv(volatility_info, file_path)            
         #Calculate the average of the 5 lowest lows over the last 30 days
        informative['rolling_lowest_lows'] = informative['low'].rolling(window=30).apply(lambda x: pd.Series(x).nsmallest(4).mean(), raw=True)
            # # Calculate the average of the 4 highest highs over the last 30 days
        informative['rolling_highest_highs'] = informative['high'].rolling(window=30).apply(lambda x: pd.Series(x).nlargest(4).mean(), raw=True)
            
            
        # # Check if there are 2+ outliers which are equal to or less than the current price over the last 45 days
        informative['rolling_max_last'] = informative['close'].rolling(window=45).max()


        # # Check if the Sell price is <5% below the Buy price
        # Check if the Sell price is less than the threshold

        # Calculate the threshold for high high values
        threshold = 1.25 * informative['rolling_highest_highs']

        # Check if any high high value exceeds the threshold
        informative['high_high_exceeds_threshold'] = informative['high'] > threshold

            # Use the next highest high value if the threshold is exceeded
        informative['rolling_highest_highs'].mask(informative['high_high_exceeds_threshold'], inplace=True)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)


        # Calculate the buy price as 0.5% higher than the average lowest low
        dataframe['buy_price'] = dataframe['rolling_lowest_lows_1d'] * 1.04

        #use close price for backtesting  
        # # Check if the current price is lower than the average lowest low
        dataframe['below_avg_low'] = dataframe['close'] < dataframe['rolling_lowest_lows_1d']
        
            # Calculate the sell price as 11% lower than the average high highs
        dataframe['sell_price'] = dataframe['rolling_highest_highs_1d'] * 0.92

            # Calculate the sell signal 
        dataframe['sell_signal'] = dataframe['rolling_highest_highs_1d'] >= dataframe['sell_price']



        
        # Calculate the difference between the 'current' price and the rolling maximum
        dataframe['close_diff_rolling_max'] = dataframe['close'] - dataframe['rolling_max_last_1d']





        return dataframe  







    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                     timeframe=self.timeframe   )
            # if side == 'long':
            new_entryprice = dataframe.iloc[-1].squeeze()
            entry_price = new_entryprice['buy_price']  # Customize for long entry
            print(f"Custom Buy Calculated : {new_entryprice['buy_price']}")
            # elif side == 'short':                
                # new_entryprice = dataframe.iloc[-1].squeeze()
                # entry_price = new_entryprice['sell_price']  # Set Sell price
                # print(f"Custom Sell Calculated : {new_entryprice['sell_price']}")
            return entry_price
    def custom_exit_price(self, pair: str, trade: Trade,
                            current_time: datetime, proposed_rate: float,
                            current_profit: float, exit_tag: Optional[str], **kwargs) -> float:

            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                    timeframe=self.timeframe)
            new_entryprice = dataframe.iloc[-1].squeeze()

            return new_entryprice['sell_price']
 



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        # pair_name = metadata['pair'].replace('/', '_')
        # os.makedirs(self.config['exchange']['name'], exist_ok=True)
        # file_path = os.path.join(self.config['exchange']['name'], f'{pair_name}.csv')
        # dataframe.to_csv(file_path, index=False)
        dataframe.loc[
            (
                # check if the current price is lower than the average lowest low
                (dataframe['below_avg_low']) & 
                # (dataframe['buy_price'] > dataframe['sell_price']) &
                (dataframe['volatility_scores_1d'] >= 4 ) 
            ),
            ['enter_long' , 'enter_tag']] = (1, 'ema_above_buy_long')
        

        return dataframe


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if trade.enter_tag == 'ema_above_buy_long' and last_candle['rolling_highest_highs_1d'] > last_candle['sell_price']:
            return 'sell_signal_ema'
        return None
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        return dataframe
