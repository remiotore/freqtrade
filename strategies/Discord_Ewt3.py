from datetime import datetime
from datetime import timedelta

import talib.abstract as ta
from finta import TA as fta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter
from freqtrade.strategy import IStrategy, merge_informative_pair

#   --------------------------------------------------------------------------------
#   Author: rextea      2021/04/28     Version: 0.01
#   --------------------------------------------------------------------------------
#   Simple strategy based on Elliot Wave Theory
#   https://www.tradingview.com/script/uculwCTj-Indicator-ElliotWave-Oscillator-EWO/
#
#
#   Posted on Freqtrade discord channel: https://discord.gg/Xr4wUYc6
from freqtrade.strategy import IntParameter


def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


class Ewt3(IStrategy):
    minimal_roi = {
        "0": 0.31
    }

    stoploss = -0.1

    timeframe = '5m'
    informative_timeframe = '15m'

    sell_profit_only = False

    process_only_new_candles = True
    startup_candle_count: int = 100

    # trailing_stop = True
    # trailing_stop_positive = 0.05
    # trailing_stop_positive_offset = 0.20
    # trailing_only_offset_is_reached = True

    # buy_EWO_val = DecimalParameter(0, 3.00, default=0.75, space='buy', optimize=True, load=True)
    # sell_EWO_val = DecimalParameter(-3.00, 0, default=-2.0, space='sell', optimize=True, load=True)
    buy_EWO = DecimalParameter(0, 3.00, default=0.566, space='buy', optimize=True, load=True)
    buy_EWO_low = DecimalParameter(-20.0, -5.0, default=-7.0, space='buy', optimize=True, load=True)
    sell_EWO = DecimalParameter(-3.00, 0, default=-0.396, space='sell', optimize=True, load=True)
    buy_sma1_len = IntParameter(5, 15, default=10, space='buy', optimize=True, load=True)
    buy_sma2_len = IntParameter(10, 40, default=40, space='buy', optimize=True, load=True)
    sell_sma1_len = IntParameter(5, 15, default=5, space='sell', optimize=True, load=True)
    sell_sma2_len = IntParameter(10, 40, default=35, space='sell', optimize=True, load=True)
    ema_len = IntParameter(10, 40, default=32, space='buy', optimize=True, load=True)

    buy_EWO.value = 1.852
    buy_EWO_low.value = -9.107
    buy_sma1_len.value = 15
    buy_sma2_len.value = 33
    ema_len.value = 18

    sell_EWO.value = -1.801
    sell_sma1_len.value = 5
    sell_sma2_len.value = 32

    custom_trade_info = {}

    # use_custom_stoploss = False
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #
    #     if self.config['runmode'].value in ('live', 'dry_run'):
    #         dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #         EWO_BUY = dataframe['EWO_BUY'].iat[-1]
    #         EWO_SELL = dataframe['EWO_SELL'].iat[-1]
    #     # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
    #     else:
    #         EWO_BUY = self.custom_trade_info[trade.pair]['EWO_BUY'].loc[current_time]['EWO_BUY']
    #         EWO_SELL = self.custom_trade_info[trade.pair]['EWO_SELL'].loc[current_time]['EWO_SELL']
    #
    #     if (current_time - timedelta(minutes=600) > trade.open_date_utc) & (current_profit < 0):
    #         if EWO_BUY < 0:
    #             return 0.01
    #     return 0.5

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        informative_pairs.append(('BTC/USDT', self.informative_timeframe))
        return informative_pairs

    def coock_indicators(self, dataframe: DataFrame, metadata: dict):
        if self.config['runmode'].value == 'hyperopt':
            for len in range(10, 41):
                dataframe[f'ema_{len}'] = fta.EMA(dataframe, period=len)

            for sma1 in range(5, 16):
                for sma2 in range(10, 41):
                    dataframe[f'EWO_{sma1}_{sma2}'] = EWO(dataframe, sma1, sma2)
        else:
            dataframe[f'ema_{self.ema_len.value}'] = fta.EMA(dataframe, period=self.ema_len.value)
            dataframe[f'EWO_{self.buy_sma1_len.value}_{self.buy_sma2_len.value}'] = EWO(dataframe,
                                                                                        self.buy_sma1_len.value,
                                                                                        self.buy_sma2_len.value)
            dataframe[f'EWO_{self.sell_sma1_len.value}_{self.sell_sma2_len.value}'] = EWO(dataframe,
                                                                                          self.sell_sma1_len.value,
                                                                                          self.sell_sma2_len.value)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}

        if self.timeframe == self.informative_timeframe:
            dataframe = self.coock_indicators(dataframe, metadata)
        else:
            assert self.dp, "DataProvider is required for multiple timeframes."

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
            informative = self.coock_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                               ffill=True)
            # don't overwrite the base dataframe's OHLCV information
            skip_columns = [(s + "_" + self.informative_timeframe) for s in
                            ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (
                not s in skip_columns) else s, inplace=True)

            # informative_btc = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.informative_timeframe)
            # informative_btc['EWO_BTC'] = EWO(informative_btc, 10, 40)
            # informative_btc['ema32_btc'] = fta.EMA(informative_btc, period=32)
            # informative_btc['btc_open'] = informative_btc['open']
            #
            # dataframe = merge_informative_pair(dataframe, informative_btc, self.timeframe, self.informative_timeframe,
            #                                    ffill=True)
            # # don't overwrite the base dataframe's OHLCV information
            # skip_columns = [(s + "_" + self.informative_timeframe) for s in
            #                 ['date', 'open', 'high', 'low', 'close', 'volume']]
            # dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (
            #     not s in skip_columns) else s, inplace=True)

        # dataframe['4h_high'] = dataframe['close'].rolling(48).max()
        # dataframe['8h_high'] = dataframe['close'].rolling(96).max()

        # if self.dp.runmode.value in ('backtest', 'hyperopt'):
        #     self.custom_trade_info[metadata['pair']]['EWO_BUY'] = dataframe[['date', 'EWO_BUY']].copy().set_index(
        #         'date')
        #     self.custom_trade_info[metadata['pair']]['EWO_SELL'] = dataframe[['date', 'EWO_SELL']].copy().set_index(
        #         'date')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ewo = f'EWO_{self.buy_sma1_len.value}_{self.buy_sma2_len.value}'
        ema = f'ema_{self.ema_len.value}'
        dataframe.loc[
            (
                    (dataframe['volume'] > 0) &
                    (
                            (
                                #     (
                                #             (dataframe['EWO_BUY'].shift(1) < 0.00) |
                                #             (dataframe['EWO_BUY'].shift(2) < 0.00) |
                                #             (dataframe['EWO_BUY'].shift(3) < 0.00)
                                #     ) &
                                    (dataframe[ewo] > self.buy_EWO.value) &
                                    (dataframe['open'] > dataframe[ema])
                            ) |
                            (dataframe[ewo] < self.buy_EWO_low.value)
                    )
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ewo = f'EWO_{self.sell_sma1_len.value}_{self.sell_sma2_len.value}'
        dataframe.loc[
            (
                    (dataframe[ewo] <= self.sell_EWO.value)
                    # & (dataframe['EWO_SELL'] > dataframe['EWO_SELL'].shift(1)) &
                    # (dataframe['EWO_SELL'].shift(1) > dataframe['EWO_SELL'].shift(2)) &
                    & (dataframe['volume'] > 0)
            ),
            'sell'
        ] = 1
        # dataframe.loc[:, 'sell'] = 0
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        # activate sell signal only when profit is above 1.5% and below -1.5%
        if sell_reason == 'sell_signal':
            if 0.02 > trade.calc_profit_ratio(rate):
                return False
            else:
                return True
        return True
