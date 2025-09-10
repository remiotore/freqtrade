



import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from datetime import datetime, timedelta
from pandas import DataFrame, Series
from freqtrade.persistence import Trade

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, IStrategy, merge_informative_pair)


import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


def williams_r(dataframe: DataFrame, period: int = 14) -> pd.Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = pd.Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100


class JT_V04(IStrategy):


    INTERFACE_VERSION = 3

    timeframe = '5m'

    can_short: bool = False


    minimal_roi = {
        "0": 0.15,
        "5": 0.10,
        "10": 0.08,
        "15": 0.06,
        "30": 0.04,
        "40": 0.03,
        "60": 0.02,
        "120": 0.005
    }







    stoploss = -0.99
    use_custom_stoploss = False






    process_only_new_candles = True

    use_exit_signal = True



    startup_candle_count: int = 60

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

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
    buy_params = { 
        "bbdelta_close": 0.00082,
        "bbdelta_tail": 0.85788,
        "close_bblower": 0.00128,
        "closedelta_close": 0.00987,
        "low_offset": 0.991,
        "rocr1_1h": 0.9346,
        "rocr_1h": 0.65666,                       
        "base_nb_candles_buy": 12,  # value loaded from strategy
        "buy_bb_delta": 0.025,  # value loaded from strategy
        "buy_bb_factor": 0.995,  # value loaded from strategy
        "buy_bb_width": 0.095,  # value loaded from strategy   
        "buy_bb_width_1h": 1.074,  # value loaded from strategy        
        "buy_cci": -116,  # value loaded from strategy                                                                 
        "buy_cci_length": 25,  # value loaded from strategy 
        "buy_closedelta": 15.0,  # value loaded from strategy                                                                                                                                                                                 
        "buy_clucha_bbdelta_close": 0.049,  # value loaded from strategy
        "buy_clucha_bbdelta_tail": 1.146,  # value loaded from strategy
        "buy_clucha_close_bblower": 0.018,  # value loaded from strategy
        "buy_clucha_closedelta_close": 0.017,  # value loaded from strategy
        "buy_clucha_rocr_1h": 0.526,  # value loaded from strategy
        "buy_ema_diff": 0.025,  # value loaded from strategy
        "buy_rmi": 49,  # value loaded from strategy
        "buy_rmi_length": 17,  # value loaded from strategy 
        "buy_roc_1h": 10,  # value loaded from strategy
        "buy_srsi_fk": 32,  # value loaded from strategy
    }


    sell_params = {
        "high_offset": 1.012,
        "high_offset_2": 1.016,
        "sell_deadfish_bb_factor": 1.089,
        "sell_deadfish_bb_width": 0.11,
        "sell_deadfish_profit": -0.107,
        "sell_deadfish_volume_factor": 1.761,
        "base_nb_candles_sell": 22,  # value loaded from strategy
        "pHSL": -0.397,  # value loaded from strategy
        "pPF_1": 0.012,  # value loaded from strategy
        "pPF_2": 0.07,  # value loaded from strategy
        "pSL_1": 0.015,  # value loaded from strategy
        "pSL_2": 0.068,  # value loaded from strategy
        "sell_bbmiddle_close": 1.09092,  # value loaded from strategy
        "sell_fisher": 0.46406,  # value loaded from strategy
        "sell_trail_down_1": 0.03,  # value loaded from strategy
        "sell_trail_down_2": 0.015,  # value loaded from strategy
        "sell_trail_profit_max_1": 0.4,  # value loaded from strategy
        "sell_trail_profit_max_2": 0.11,  # value loaded from strategy
        "sell_trail_profit_min_1": 0.1,  # value loaded from strategy
        "sell_trail_profit_min_2": 0.04,  # value loaded from strategy
    }

    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize=False)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize=False)

    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.08 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.5 ,space='sell', optimize = is_optimize_deadfish)

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)
    
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.11, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space='sell', decimals=3, optimize=False, load=True)

    
    

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        dataframe['avg_volume'] = dataframe['volume'].rolling(window=10).mean()

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']



        upperband, middleband, lowerband = ta.BBANDS(dataframe['ha_close'], timeperiod=40, nbdevup=2.0, nbdevdn=2.0, matype=0)

        dataframe['lower'] = lowerband
        dataframe['mid'] = middleband



        dataframe['bb_upperband'] = upperband
        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']
        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe['ha_close'], timeperiod=200)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)  # 能找出低点，但是相对有一些局限性。比较延迟
        
        rsi = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi"] = rsi



        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ma_sell_22'] = ta.EMA(dataframe, timeperiod=22)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['vwma'] = pta.vwma(dataframe["ha_close"], dataframe["volume"], 28*0.98)


        dataframe['close_delta'] = (dataframe['close'] - 0.990 * dataframe['bb_lowerband']) / dataframe['close']

        return dataframe

    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_time - timedelta(minutes=60) > trade.open_date_utc:
            if (current_candle["fastk"] > 75) and (current_profit > -0.01):
                return -0.001

        if current_time - timedelta(days=1) > trade.open_date_utc:
            if (current_candle["fastk"] > 75) and (current_profit > -0.05):
                return -0.001

        enter_tag = ''
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            enter_tag = trade.enter_tag
        enter_tags = enter_tag.split()

        if current_profit >= 0.05:
            return -0.005

        if current_profit > 0:
            if current_candle["fastk"] > 75:
                return -0.001

        return self.stoploss

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)
        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        
        if last_candle is not None and "rocr" in buy_tags:
            if current_profit >= 0.022:
                return 'sell dazhuan'
            
            if (current_profit > 0.1) & (current_profit < 0.4) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + 0.03)):
                return 'trail_target_1'
            elif (current_profit > 0.04) & (current_profit < 0.11) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + 0.015)):
                return 'trail_target_2'
            elif (current_profit > 0.03) & (last_candle['rsi'] > 85):
                 return 'RSI-85 target'

            if (current_profit > 0) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle['ma_sell_22'] * 1.012)) &  (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell signal2' 
            
            if ((current_profit < self.sell_deadfish_profit.value)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value)
                and (last_candle['cmf'] < 0.0)
            ):
                return "sell_stoploss_deadfish"

        if "close_delta" in buy_tags and last_candle is not None:



            if (current_time - timedelta(minutes=60) > trade.open_date_utc) and last_candle['enter_tag'] == "close_delta":
                return "sell chuanxi"



            if current_profit >= 0.0065:
                return 'sell xiaozhuan'








    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
        (       
                (dataframe['rocr'] < 0.95) &
                (dataframe['ema_fast'] <= 0.988*dataframe['bb_lowerband'])
        ),
        ['enter_long', 'enter_tag']
        ] = (1, "rocr")

        dataframe.loc[
        (       
            (dataframe['rocr'] < 0.965) &   # 0.985
            (dataframe['ha_high'] < dataframe['vwma']) &
            (dataframe['close_delta'] >= 0) &
            (dataframe['close_delta'] <= 0.0035)


        ),
        ['enter_long', 'enter_tag']
        ] = (1, "close_delta")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rocr"] > 1.02) &
            ((dataframe['ema_fast']) >= dataframe['bb_upperband']) &
            (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, "tag1")

        dataframe.loc[
            (dataframe["rocr"] > 1.06) &
            (dataframe["low"] > dataframe['vwma'])&
            (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'tag2')
        return dataframe

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        if pair == "BTC/USDT" or pair == "ETH/USDT":
            return proposed_rate
        return proposed_rate*0.99

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if entry_tag == "rocr":
            return self.wallets.get_total_stake_amount() * 0.75
        else:
            return self.wallets.get_total_stake_amount() / self.config['max_open_trades']


def chaikin_money_flow(dataframe, n=20, fillna=False):
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')
    
