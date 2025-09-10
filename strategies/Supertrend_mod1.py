# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
from freqtrade.persistence import Trade
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import numpy as np
from datetime import datetime

def supertrend(dataframe, multiplier=3, period=10):
    """
    Supertrend Indicator
    adapted for freqtrade
    from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """
    df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = df['TR'].ewm(alpha=1 / period).mean()

    # atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    # Compute basic upper and lower bands
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < df['final_lb'].iat[i] else 0.00

    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    # df.to_csv('user_data/Supertrend.csv')
    return DataFrame(index=df.index, data={
        'ST': df[st],
        'STX': df[stx]
    })


class Supertrend_mod1(IStrategy):

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #     "0": 0.1,
    #     "2880": 0.01
    # }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.99

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["supertrend_3_12"] = supertrend(dataframe, 3, 12)["STX"]
        dataframe["supertrend_1_10"] = supertrend(dataframe, 1, 10)["STX"]
        dataframe["supertrend_2_11"] = supertrend(dataframe, 2, 11)["STX"]

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe["supertrend_3_12"] == "up") &
                        (dataframe["supertrend_1_10"] == "up") &
                        (dataframe["supertrend_2_11"] == "up")
                )
            ),
            'buy'] = 1
        return dataframe
    
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # print(dir(dataframe))
        last_candle = dataframe.iloc[-1].squeeze()
        candle_before_last = dataframe.iloc[-2].squeeze()
        candle_before_before_last = dataframe.iloc[-3].squeeze()
        trade_duration = (current_time - trade.open_date_utc).total_seconds()

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag


        if buy_tag == 'b1ewo' and current_profit >= 0.08:
            return 'b1sell'
        elif buy_tag == 'b1ewo' and current_profit >= 0.04 and trade_duration >= 3*24*3600:
            return 'tb1sell'
        elif current_profit  >= 0.06 and buy_tag != 'b1ewo': 
            return 'RR6'
        elif buy_tag != 'b1ewo' and last_candle['close'] < candle_before_last['close'] and candle_before_last['close'] > candle_before_before_last['close'] and current_profit >= 0.05:
             return 'pullback'
        # elif current_profit >= 0.05: 
        #     return 'RR5'
        # elif current_profit >= 0.04: 
        #     return 'RR4'
        # elif last_candle['rsi'] > 80 and current_profit >= 0.06:
        #     return 'RSI'

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe.loc[
        #     (
        #         (
        #                 (dataframe["supertrend_3_12"] == "down") &
        #                 (dataframe["supertrend_1_10"] == "down") &
        #                 (dataframe["supertrend_2_11"] == "down")
        #          )
        #     ),
        #     'sell'] = 1
        return dataframe
