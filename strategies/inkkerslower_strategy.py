import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter
from functools import reduce
import pandas as pd










































class inkkerslower_strategy(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.028,         # I feel lucky!
        "10": 0.018,
        "40": 0.005,
        "180": 0.018,        # We're going up?
    }


    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_entry_signal = False

    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    use_custom_stoploss = True

    process_only_new_candles = True

    startup_candle_count: int = 200

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    buy_params = {


        "buy_condition_0_enable": True,
    }

    sell_params = {


        "sell_condition_0_enable": True,
    }






    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:

        return True


    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        return False


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        if (current_profit > 0):
            return 0.99
        else:
            trade_time_50 = trade.open_date_utc + timedelta(minutes=50)



            if (current_time > trade_time_50):

                try:
                    number_of_candle_shift = int((current_time - trade_time_50).total_seconds() / 300)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()

                    if candle['rsi_1h'] < 30:
                        return 0.99

                    if candle['close'] > candle['ema_200']:
                        if current_rate * 1.025 < candle['open']:
                            return 0.01 

                    if current_rate * 1.015 < candle['open']:
                        return 0.01

                except IndexError as error:

                    return 0.1

        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h['ema_25'] = ta.EMA(informative_1h, timeperiod=25)
        informative_1h['ema_10'] = ta.EMA(informative_1h, timeperiod=10)
        informative_1h['ema_5'] = ta.EMA(informative_1h, timeperiod=5)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)






        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:






        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_75'] = ta.EMA(dataframe, timeperiod=75)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)

        ccilength = 5
        wmalength = 9
        dataframe['cci'] = ta.CCI(dataframe['high'], dataframe['low'], dataframe['close'], window=ccilength, constant=0.015, fillna=False)
        dataframe['v11'] = (dataframe['cci'].divide(4)).multiply(0.1)
        dataframe['v21'] = ta.WMA(dataframe['v11'], window=wmalength)
        dataframe['result1'] = np.exp(dataframe['v21'].multiply(2))
        dataframe['iftcombo'] = (dataframe['result1'].subtract(1)).divide(dataframe['result1'].add(1))







        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] =  (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3

        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)



        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe
























    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def get_rsi(close, lookback):
        scr = close.diff()
        up = []
        down = []
        for i in range(len(scr)):
            if scr[i] < 0:
                up.append(0)
                down.append(scr[i])
            else:
                up.append(scr[i])
                down.append(0)
        up_series = pd.Series(up)
        down_series = pd.Series(down).abs()
        up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
        down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
        rs = up_ewm/down_ewm
        rsi = 100-(100/(1+rs))
        rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
        rsi_df = rsi_df.dropna()
        return rsi_df[3:]




    def implement_rsi_strategy(prices, rsi):
        buy_price = []
        sell_price = []
        rsi_signal = []
        signal = 0

        for i in range(len(rsi)):
            if rsi[i-1] > 30 and rsi[i] < 30:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    rsi_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(0)
            elif rsi[i-1] < 70 and rsi[i] > 70:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    rsi_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)

        return buy_price, sell_price, rsi_signal


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            (


                (dataframe['close'] > dataframe['ema_200']) &

                (dataframe['ema_5'] > dataframe['ema_10']) &

                (dataframe['ema_5'] > dataframe['ema_25']) &

                (dataframe['iftcombo'] < -0.6) &

                (dataframe['hist'] > 0) &
                (dataframe['hist'].shift(2) < 0) &

                (dataframe['crsi'] < 35) &

                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []









        conditions.append(
            (

                (dataframe['close'] < dataframe['ema_200']) &

                (dataframe['ema_5'] < dataframe['ema_10']) &
                (dataframe['ema_5'].shift(2) > dataframe['ema_10'].shift(2)) &

                (dataframe['ema_5'] < dataframe['ema_25']) &
                (dataframe['ema_5'].shift(2) > dataframe['ema_25'].shift(2)) &

                (dataframe['iftcombo'] > 0.6) &

                (dataframe['hist'] < 0) &
                (dataframe['hist'].shift(2) > 0) &

                (dataframe['crsi'] > 64.5) &

                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        return dataframe