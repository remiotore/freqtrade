# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
import asyncio
# --------------------------------
import json
from websocket import create_connection
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter
from datetime import datetime, timedelta
from freqtrade.optimize.space import SKDecimal
from skopt.space import Categorical, Dimension, Integer
from typing import Dict, List
import math

# Tip and Dip
class TAD(IStrategy):
    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            """
            Create a ROI table.

            Generates the ROI table that will be used by Hyperopt.
            You may override it in your custom Hyperopt class.
            """
            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6'] + params['roi_p7'] + params['roi_p8']
            roi_table[params['roi_t8']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6'] + params['roi_p7']
            roi_table[params['roi_t8'] + params['roi_t7']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3']] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2']] = params['roi_p1']
            roi_table[params['roi_t8'] + params['roi_t7'] + params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0
            
            return roi_table
        
        @staticmethod
        def roi_space() -> List[Dimension]:
            """
            Create a ROI space.

            Defines values to search for each ROI steps.

            This method implements adaptive roi hyperspace with varied
            ranges for parameters which automatically adapts to the
            timeframe used.

            It's used by Freqtrade by default, if no custom roi_space method is defined.
            """

            # Default scaling coefficients for the roi hyperspace. Can be changed
            # to adjust resulting ranges of the ROI tables.
            # Increase if you need wider ranges in the roi hyperspace, decrease if shorter
            # ranges are needed.
            roi_t_alpha = 1.0
            roi_p_alpha = 1.0

            timeframe_min = 1

            # We define here limits for the ROI space parameters automagically adapted to the
            # timeframe used by the bot:
            #
            # * 'roi_t' (limits for the time intervals in the ROI tables) components
            #   are scaled linearly.
            # * 'roi_p' (limits for the ROI value steps) components are scaled logarithmically.
            #
            # The scaling is designed so that it maps exactly to the legacy Freqtrade roi_space()
            # method for the 5m timeframe.
            roi_t_scale = timeframe_min / 1
            roi_p_scale = math.log1p(timeframe_min) / math.log1p(5)
            roi_limits = {
                'roi_t1_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t1_max': int(600 * roi_t_scale * roi_t_alpha),
                'roi_t2_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t2_max': int(450 * roi_t_scale * roi_t_alpha),
                'roi_t3_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t3_max': int(300 * roi_t_scale * roi_t_alpha),
                'roi_t4_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t4_max': int(250 * roi_t_scale * roi_t_alpha),
                'roi_t5_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t5_max': int(200 * roi_t_scale * roi_t_alpha),
                'roi_t6_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t6_max': int(150 * roi_t_scale * roi_t_alpha),
                'roi_t7_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t7_max': int(100 * roi_t_scale * roi_t_alpha),
                'roi_t8_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_t8_max': int(50 * roi_t_scale * roi_t_alpha),
                'roi_t8_min': int(1 * roi_t_scale * roi_t_alpha),
                'roi_p1_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p1_max': 0.075 * roi_p_scale * roi_p_alpha,
                'roi_p2_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p2_max': 0.10 * roi_p_scale * roi_p_alpha,
                'roi_p3_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p3_max': 0.125 * roi_p_scale * roi_p_alpha,
                'roi_p4_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p4_max': 0.15 * roi_p_scale * roi_p_alpha,
                'roi_p5_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p5_max': 0.175 * roi_p_scale * roi_p_alpha,
                'roi_p6_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p6_max': 0.20 * roi_p_scale * roi_p_alpha,
                'roi_p7_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p7_max': 0.25 * roi_p_scale * roi_p_alpha,
                'roi_p8_min': 0.002 * roi_p_scale * roi_p_alpha,
                'roi_p8_max': 0.30 * roi_p_scale * roi_p_alpha,
            }
            p = {
                'roi_t1': roi_limits['roi_t1_min'],
                'roi_t2': roi_limits['roi_t2_min'],
                'roi_t3': roi_limits['roi_t3_min'],
                'roi_t4': roi_limits['roi_t4_min'],
                'roi_t5': roi_limits['roi_t5_min'],
                'roi_t6': roi_limits['roi_t6_min'],
                'roi_t7': roi_limits['roi_t7_min'],
                'roi_t8': roi_limits['roi_t8_min'],
                'roi_p1': roi_limits['roi_p1_min'],
                'roi_p2': roi_limits['roi_p2_min'],
                'roi_p3': roi_limits['roi_p3_min'],
                'roi_p4': roi_limits['roi_p4_min'],
                'roi_p5': roi_limits['roi_p5_min'],
                'roi_p6': roi_limits['roi_p6_min'],
                'roi_p7': roi_limits['roi_p7_min'],
                'roi_p8': roi_limits['roi_p8_min'],
            }
            p = {
                'roi_t1': roi_limits['roi_t1_max'],
                'roi_t2': roi_limits['roi_t2_max'],
                'roi_t3': roi_limits['roi_t3_max'],
                'roi_t4': roi_limits['roi_t4_max'],
                'roi_t5': roi_limits['roi_t5_max'],
                'roi_t6': roi_limits['roi_t6_max'],
                'roi_t7': roi_limits['roi_t7_max'],
                'roi_t8': roi_limits['roi_t8_max'],
                'roi_p1': roi_limits['roi_p1_max'],
                'roi_p2': roi_limits['roi_p2_max'],
                'roi_p3': roi_limits['roi_p3_max'],
                'roi_p4': roi_limits['roi_p4_max'],
                'roi_p5': roi_limits['roi_p5_max'],
                'roi_p6': roi_limits['roi_p6_max'],
                'roi_p7': roi_limits['roi_p7_max'],
                'roi_p8': roi_limits['roi_p8_max'],
            }

            return [
                Integer(roi_limits['roi_t1_min'], roi_limits['roi_t1_max'], name='roi_t1'),
                Integer(roi_limits['roi_t2_min'], roi_limits['roi_t2_max'], name='roi_t2'),
                Integer(roi_limits['roi_t3_min'], roi_limits['roi_t3_max'], name='roi_t3'),
                Integer(roi_limits['roi_t4_min'], roi_limits['roi_t4_max'], name='roi_t4'),
                Integer(roi_limits['roi_t5_min'], roi_limits['roi_t5_max'], name='roi_t5'),
                Integer(roi_limits['roi_t6_min'], roi_limits['roi_t6_max'], name='roi_t6'),
                Integer(roi_limits['roi_t7_min'], roi_limits['roi_t7_max'], name='roi_t7'),
                Integer(roi_limits['roi_t8_min'], roi_limits['roi_t8_max'], name='roi_t8'),
                SKDecimal(roi_limits['roi_p1_min'], roi_limits['roi_p1_max'], decimals=3, name='roi_p1'),
                SKDecimal(roi_limits['roi_p2_min'], roi_limits['roi_p2_max'], decimals=3, name='roi_p2'),
                SKDecimal(roi_limits['roi_p3_min'], roi_limits['roi_p3_max'], decimals=3, name='roi_p3'),
                SKDecimal(roi_limits['roi_p4_min'], roi_limits['roi_p4_max'], decimals=3, name='roi_p4'),
                SKDecimal(roi_limits['roi_p5_min'], roi_limits['roi_p5_max'], decimals=3, name='roi_p5'),
                SKDecimal(roi_limits['roi_p6_min'], roi_limits['roi_p6_max'], decimals=3, name='roi_p6'),
                SKDecimal(roi_limits['roi_p7_min'], roi_limits['roi_p7_max'], decimals=3, name='roi_p7'),
                SKDecimal(roi_limits['roi_p8_min'], roi_limits['roi_p8_max'], decimals=3, name='roi_p8'),
            ]
    
    
    # Sell signal
    #use_sell_signal = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    #ignore_roi_if_buy_signal = False
    # Custom stoploss
    ws = create_connection("wss://stream.binance.com:9443/ws/!ticker@arr")
    use_custom_stoploss = False
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    BTC_1m_param_bottom = DecimalParameter(-1.5, 0, default=-0.92, space='buy', decimals=2, optimize=True)
 
    #stoploss_time = IntParameter(45, 480, default=120, space='buy', optimize=True)
    #stoploss_custom = DecimalParameter(-0.2, -0.05, default=-0.05, space='buy', decimals=2, optimize=True)
    #sell_custom_stoploss_1 = DecimalParameter(-0.15, -0.03, default=-0.05, space='sell', decimals=2, optimize=False, load=True)
    #Hyperopt

    # ROI table:
    minimal_roi = {
        "0": 0.196,
        "44": 0.173,
        "75": 0.141,
    }

    # Stoploss:
    stoploss = -0.184

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.161
    trailing_stop_positive_offset = 0.214
    trailing_only_offset_is_reached = True
    tickerData = pd.DataFrame()
    
    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '5m') for pair in pairs]
        informative_pairs += [(pair, '1h') for pair in pairs]
        return informative_pairs
        
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        # Get the informative pair
        informative_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='5m')
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        
        # Get the 14 day rsi
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        dataframe = merge_informative_pair(dataframe, informative_5m, self.timeframe, '5m', ffill=True)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
        
        #dont trade on sundays til monday 6am
        dataframe['dontbuy'] = ((dataframe['date'].dt.dayofweek == 6) & (dataframe['date'].dt.hour >= 6)) | ((dataframe['date'].dt.dayofweek == 0) & (dataframe['date'].dt.hour < 6))
        return dataframe
        
    def bot_loop_start(self, *kwargs) -> None:
        self.tickerData = self.getTicker()
        return
    def getTicker(self) -> DataFrame:
        #self.on_ping(message="pong!")
        result = self.ws.recv()
        tickData = pd.read_json(result)
        print (tickData)
        return tickData

    def extractValuesFromTicker(self, strippedPair: str,) -> float:
        extractedValues = self.tickerData[self.tickerData['s']==strippedPair]['c'].values
        return extractedValues

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        extractedValues = self.extractValuesFromTicker(metadata['pair'].replace('/', ''))
        print(extractedValues)
        conditions.append(
                (metadata['pair']!='BTC/USDT') &
                (dataframe['dontbuy'] == False) &
                #(self.dataframe['rsi_1h'] > self.pair_buy_rsi_1h_param_top.top.value) &
                (dataframe['volume'] > 0)
        )
        
        
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1
        #print(dataframe.tail(10))
        #dataframe.to_csv('Dataframe_Export.csv')
        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                False &
                #(qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                #(dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
                #(dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
            
        return dataframe








