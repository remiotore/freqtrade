
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone

from typing import Dict, List
import numpy as np



class cryptohassle(IStrategy):
    """

    author@: Sp0ngeB0bUK
    Title:  Crypto Hassle 
    Version: 0.1

    Heikin Ashi Candles - SSL Channel, Momentum cross supported by MACD
    
    """
    
    
   
    minimal_roi = {
        "0": 0.50,

        
    }

    stoploss = -0.20

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.07
    trailing_only_offset_is_reached = True

    ticker_interval = '1h'

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    plot_config = {
        'main_plot': {


            'ha_ema9': {'color': 'green'},
            'ha_ema20': {'color': 'red'}
        },
        'subplots': {

            "ADX": {
                'ha_adx': {'color': 'blue'}
            }
        }
    }
    
    
       
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)













        dataframe['ha_mom'] = ta.MOM(heikinashi, timeperiod=14)
        dataframe['ha_mom_cross_above'] = qtpylib.crossed_above(dataframe['ha_mom'],0)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']  
        dataframe['ha_high'] = heikinashi['high'] 
        dataframe['ha_low'] = heikinashi['low']

        macd = ta.MACD(heikinashi)
        dataframe['ha_macd'] = macd['macd']
        dataframe['ha_macdsignal'] = macd['macdsignal']
        dataframe['ha_macdhist'] = macd['macdhist']
        dataframe['ha_macd_cross_above'] = qtpylib.crossed_above(dataframe['ha_macd'],dataframe['ha_macdsignal'])

        def SSLChannels(dataframe, length = 10, mode='sma'):
            """
            Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
            Author: xmatthias
            Pinescript Author: ErwinBeckers
            SSL Channels.
            Average over highs and lows form a channel - lines "flip" when close crosses either of the 2 lines.
            Trading ideas:
                * Channel cross
                * as confirmation based on up > down for long
                MC - MODIFIED FOR HA CANDLES
            """
            if mode not in ('sma'):
                raise ValueError(f"Mode {mode} not supported yet")
            df = dataframe.copy()
            if mode == 'sma':
                df['smaHigh'] = df['ha_high'].rolling(length).mean()
                df['smaLow'] = df['ha_low'].rolling(length).mean()
            df['hlv'] = np.where(df['ha_close'] > df['smaHigh'], 1, np.where(df['ha_close'] < df['smaLow'], -1, np.NAN))
            df['hlv'] = df['hlv'].ffill()
            df['ha_sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
            df['ha_sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
            return df['ha_sslDown'], df['ha_sslUp']
        ssl = SSLChannels(dataframe, 10)
        dataframe['ha_sslDown'] = ssl[0]
        dataframe['ha_sslUp'] = ssl[1]
        dataframe['ha_ssl_cross_above'] = qtpylib.crossed_above(dataframe['ha_sslUp'],dataframe['ha_sslDown'])
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                    (dataframe['ha_ssl_cross_above'].rolling(5).apply(lambda x: x.any(), raw=False) == 1) &

                    (dataframe['ha_mom_cross_above'].rolling(5).apply(lambda x: x.any(), raw=False) == 1) &

                    (dataframe['ha_macd_cross_above'].rolling(5).apply(lambda x: x.any(), raw=False) == 1) &

                    (dataframe['volume'] > 1000)
                    
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                
                qtpylib.crossed_below(dataframe['ha_sslUp'],dataframe['ha_sslDown']) &
                (dataframe['volume'] > 0)
                              
            ),
        'sell'] = 1
        return dataframe