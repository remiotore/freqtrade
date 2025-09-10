
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.strategy import DecimalParameter, IntParameter

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class BBRSITV_239(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        "ewo_high": 4.86,
        "for_ma_length": 22,
        "for_sigma": 1.74,
    }

    sell_params = {
        "for_ma_length_sell": 65,
        "for_sigma_sell": 1.895,
        "rsi_high": 72,
    }

    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25  # value loaded from strategy

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    process_only_new_candles = True
    startup_candle_count = 30

    protections = [














        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.14
        },
    ]

    ewo_high = DecimalParameter(0, 7.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    for_sigma = DecimalParameter(0, 10.0, default=buy_params['for_sigma'], space='buy', optimize=True)
    for_sigma_sell = DecimalParameter(0, 10.0, default=sell_params['for_sigma_sell'], space='sell', optimize=True)
    rsi_high = IntParameter(60, 100, default=sell_params['rsi_high'], space='sell', optimize=True)
    for_ma_length = IntParameter(5, 80, default=buy_params['for_ma_length'], space='buy', optimize=True)
    for_ma_length_sell = IntParameter(5, 80, default=sell_params['for_ma_length_sell'], space='sell', optimize=True)

    timeframe = '5m'

    fast_ewo = 50
    slow_ewo = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:





        src = 'close'

        for_rsi = 14



        for_mult = 2

        for_sigma = 0.1



        dataframe['rsi'] = ta.RSI(dataframe[src], for_rsi)
        if self.config['runmode'].value == 'hyperopt':
            for for_ma in range(5, 81):

                dataframe[f'basis_{for_ma}'] = ta.EMA(dataframe['rsi'], for_ma)

                dataframe[f'dev_{for_ma}'] = ta.STDDEV(dataframe['rsi'], for_ma)









        else:
            dataframe[f'basis_{self.for_ma_length.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'basis_{self.for_ma_length_sell.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length_sell.value)

            dataframe[f'dev_{self.for_ma_length.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'dev_{self.for_ma_length_sell.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length_sell.value)



        h1 = 70

        h2 = 30






















        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (






                qtpylib.crossed_below(dataframe['rsi'], (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value))) &
                (dataframe['EWO'] >  self.ewo_high.value) &
                (dataframe['volume'] > 0)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['rsi'] > self.rsi_high.value) |




                    qtpylib.crossed_above(dataframe['rsi'], dataframe[f'basis_{self.for_ma_length_sell.value}'] + ((dataframe[f'dev_{self.for_ma_length_sell.value}'] * self.for_sigma_sell.value)))
                ) &
                (dataframe['volume'] > 0)

            ),
            'sell'] = 1
        return dataframe


class BBRSITV1(BBRSITV_239):
    """
    2021-07-01 00:00:00 -> 2021-09-28 00:00:00 | Max open trades : 4
============================================================================= STRATEGY SUMMARY =============================================================================
|              Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |              Drawdown |
|-----------------------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------+-----------------------|
|         Elliotv8_08SL |    906 |           0.92 |         832.19 |         19770.304 |         659.01 |        0:38:00 |   717     0   189  79.1 | 2020.917 USDT  79.84% |
| SMAOffsetProtectOptV1 |    417 |           1.33 |         555.91 |          8423.809 |         280.79 |        1:44:00 |   300     0   117  71.9 | 1056.072 USDT  61.08% |
|               BBRSITV |    309 |           1.10 |         340.17 |          3869.800 |         128.99 |        2:53:00 |   223     0    86  72.2 |  261.984 USDT  25.84% |
============================================================================================================================================================================
    """
    INTERFACE_VERSION = 2

    buy_params = {
        "ewo_high": 4.964,
        "for_ma_length": 12,
        "for_sigma": 2.313,
    }

    sell_params = {
        "for_ma_length_sell": 78,
        "for_sigma_sell": 1.67,
        "rsi_high": 60,
    }

    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25  # value loaded from strategy

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy



class BBRSITV1_SL20(BBRSITV1):

    stoploss = -0.20  # value loaded from strategy



class BBRSITV1_SL18(BBRSITV1):

    stoploss = -0.18  # value loaded from strategy


class BBRSITV1_SL15(BBRSITV1):

    stoploss = -0.15  # value loaded from strategy


class BBRSITV1_SL12(BBRSITV1):

    stoploss = -0.12  # value loaded from strategy


class BBRSITV1_SL10(BBRSITV1):

    stoploss = -0.1  # value loaded from strategy


class BBRSITV1_SL08(BBRSITV1):

    stoploss = -0.08  # value loaded from strategy


class BBRSITV1_SL06(BBRSITV1):

    stoploss = -0.06  # value loaded from strategy


class BBRSITV2(BBRSITV_239):
    """
    2021-07-01 00:00:00 -> 2021-09-28 00:00:00 | Max open trades : 4
============================================================================= STRATEGY SUMMARY =============================================================================
|              Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |              Drawdown |
|-----------------------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------+-----------------------|
|         Elliotv8_08SL |    906 |           0.92 |         832.19 |         19770.304 |         659.01 |        0:38:00 |   717     0   189  79.1 | 2020.917 USDT  79.84% |
| SMAOffsetProtectOptV1 |    417 |           1.33 |         555.91 |          8423.809 |         280.79 |        1:44:00 |   300     0   117  71.9 | 1056.072 USDT  61.08% |
|               BBRSITV |    486 |           1.11 |         537.58 |          7689.862 |         256.33 |        5:01:00 |   287     0   199  59.1 | 1279.461 USDT  75.45% |
============================================================================================================================================================================
    """

    buy_params = {
        "ewo_high": 4.85,
        "for_ma_length": 11,
        "for_sigma": 2.066,
    }

    sell_params = {
        "for_ma_length_sell": 61,
        "for_sigma_sell": 1.612,
        "rsi_high": 87,
    }

    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25  # value loaded from strategy

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy


class BBRSITV3(BBRSITV_239):
    """

    2021-07-01 00:00:00 -> 2021-09-28 00:00:00 | Max open trades : 4
    ============================================================================== STRATEGY SUMMARY =============================================================================
    |              Strategy |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |               Drawdown |
    |-----------------------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------+------------------------|
    |         Elliotv8_08SL |    906 |           0.92 |         832.19 |         19770.304 |         659.01 |        0:38:00 |   717     0   189  79.1 | 2020.917 USDT   79.84% |
    | SMAOffsetProtectOptV1 |    417 |           1.33 |         555.91 |          8423.809 |         280.79 |        1:44:00 |   300     0   117  71.9 | 1056.072 USDT   61.08% |
    |               BBRSITV |    627 |           1.14 |         715.85 |         12998.605 |         433.29 |        5:35:00 |   374     0   253  59.6 | 2294.408 USDT  100.60% |
    ============================================================================================================================================================================="""
    INTERFACE_VERSION = 2

    buy_params = {
        "ewo_high": 4.86,
        "for_ma_length": 22,
        "for_sigma": 1.74,
    }

    sell_params = {
        "for_ma_length_sell": 65,
        "for_sigma_sell": 1.895,
        "rsi_high": 72,
    }

    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25  # value loaded from strategy

    trailing_stop = True
    trailing_stop_positive = 0.078
    trailing_stop_positive_offset = 0.095
    trailing_only_offset_is_reached = False