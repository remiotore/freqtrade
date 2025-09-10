# --- Do not remove these libs ---
from freqtrade.strategy import (BooleanParameter, DecimalParameter, IntParameter, IStrategy,RealParameter)
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime

# --------------------------------

class CustomStrategy(IStrategy):
    # Buy hyperspace params:
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_min_fan_magnitude_gain": 1.002,
        "buy_volume_multiplier": 1.2,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_trend_indicator": "trend_close_2h",
    }

    # ROI table:
    # minimal_roi = {
    #     "0": 0.05,
    #     "10": 0.03,
    #     "40": 0.02,
    #     "80": 0.01,
    #     "200": 0,
    # }
    minimal_roi = {
        "0": 0.079,
        "15": 0.047,
        "41": 0.032,
        "114": 0.11,
        "180": 0.007,
        "420": 0.001
    }
    
    stoploss = -0.32  # Tighter initial stop-loss
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True
    timeframe = "5m"
    startup_candle_count = 140
    can_short = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    min_leverage = 1.5
    max_leverage = 10
    rsi_period = 14 
    sharpe_period = 120 

    process_only_new_candles = True
    position_adjustment_enabled = True
    use_exit_signal = True
    exit_profit_only = True
    # ignore_roi_if_entry_signal = True

    plot_config = {
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
            },
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_2h': {'color': '#E3FF33'},
        },
        'subplots': {
            'fan_magnitude': {'fan_magnitude': {}},
            'fan_magnitude_gain': {'fan_magnitude_gain': {}},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Heikin Ashi candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        # Trend indicators
        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)

        dataframe['fan_magnitude'] = dataframe['trend_close_5m'] / dataframe['trend_close_2h']
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        # Ichimoku
        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']

        # Volume filter
        dataframe['vol_ma'] = dataframe['volume'].rolling(20).mean()

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Calculer le RSI (Relative Strength Index)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        # Calculer le TEMA (Triple Exponential Moving Average)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        
        # Calculate Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_middleband'] = bollinger['middleband']

        # # Calculer le ratio de Sharpe 
        # dataframe['sharpe_ratio'] = self.calculate_sharpe_ratio(dataframe, self.sharpe_period)
        
        
        return dataframe 
    
    # def calculate_sharpe_ratio(self, dataframe: DataFrame, period: int) -> DataFrame: 
        
    #     returns = dataframe['close'].pct_change() 
    #     sharpe_ratio = returns.rolling(window=period).mean() / returns.rolling(window=period).std() 
    #     return sharpe_ratio 

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Price above Ichimoku Cloud
        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        # Fan magnitude conditions
        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'])
        conditions.append(dataframe['fan_magnitude'] > 1)

        # Sufficient volume
        conditions.append(dataframe['volume'] > dataframe['vol_ma'] * self.buy_params['buy_volume_multiplier'])

        # Avoid flat markets
        conditions.append(dataframe['atr'] > dataframe['atr'].rolling(20).mean())

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
            
        dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
            (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['enter_long', 'enter_tag']] = (1, 'rsi_cross')
        # ['enter_short', 'enter_tag']] = (1, 'rsi_cross')

        dataframe.loc[
        (
            (qtpylib.crossed_below(dataframe['rsi'], 70)) &  # Signal: RSI crosses below 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['enter_short', 'enter_tag']] = (1, 'rsi_cross')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Cross below sell trend indicator
        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        

         # Paramètres du levier dynamique 
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe) 
        atr = dataframe['atr'].iloc[-1] 
        # sharpe_ratio = dataframe['sharpe_ratio'].iloc[-1] 
        # rsi = dataframe['rsi'].iloc[-1] 
        
        # # Calculer le levier basé sur l'ATR, le ratio de Sharpe et le RSI 
        dynamic_leverage = self.min_leverage + (max_leverage - self.min_leverage) * (atr / atr.max()) 
        
        # if sharpe_ratio > 1: 
        #     dynamic_leverage *= 1.5  # Augmenter le levier si le ratio de Sharpe est supérieur à 1 
        # elif sharpe_ratio < 0: 
        #     dynamic_leverage *= 0.5  # Réduire le levier si le ratio de Sharpe est inférieur à 0 
        
        # if rsi > 70: 
        #     dynamic_leverage *= 0.7  # Réduire le levier si le RSI indique une condition de surachat 
        # elif rsi < 30: 
        #     dynamic_leverage *= 2.3  # Augmenter le levier si le RSI indique une condition de survente 

        if proposed_leverage > 1.0:
            dynamic_leverage *= 1.6  # Augmenter le levier si le levier proposé est supérieur à 1
        else:
            dynamic_leverage *= 0.5  # Réduire le levier si le levier proposé est inférieur à 1

        
        return min(self.max_leverage, max(proposed_leverage, dynamic_leverage/6)) 
        

        