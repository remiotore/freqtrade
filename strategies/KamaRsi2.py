
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade


"""
Supertrend Indicator
adapted for freqtrade
from: https://github.com/freqtrade/freqtrade-strategies/issues/30
ref: https://www.tradingfuel.com/supertrend-indicator-formula-and-calculation/
ATR: https://github.com/freqtrade/technical/issues/26
"""
def supertrend(dataframe, multiplier=3, period=10):

    df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = df['TR'].ewm(alpha=1 / period).mean()

    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00 

    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return DataFrame(index=df.index, data={
        'ST' : df[st],
        'STX' : df[stx]
    })

class KamaRsi2(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    buy_params = {
        'rsi-buy-trigger': 10,
        'cci-buy-trigger': -100,
        'buy-method': 'both', # trend, rsi2, both
        'buy-price': 'ohlc4' # open, close, hl2, hlc3, ohlc4
    }

    sell_params = {
        'rsi-sell-trigger': 90,
        'cci-sell-trigger': 100,
        'sell-method': 'both', # trend, rsi2, both
        'sell-price': 'ohlc4' # open, close, hl2, hlc3, ohlc4
    }



    minimal_roi = {
        "0": 1
    }

    stoploss = -1

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01

    ignore_roi_if_buy_signal = True

    process_only_new_candles = False

    startup_candle_count: int = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ohlc4'] = ta.AVGPRICE(dataframe)
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2

        dataframe['kama-short'] = ta.KAMA(dataframe, timeperiod=5)
        dataframe['kama-long'] = ta.KAMA(dataframe, timeperiod=200)

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=5)
        dataframe['rsi'] = ta.RSI(dataframe['ohlc4'], timeperiod=2)

        dataframe['sar'] = ta.SAR(dataframe)




        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        trades = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()

        conditions = []

        if not trades:
            if params['buy-method'] == 'rsi2' or params['buy-method'] == 'both':
                conditions.append(dataframe[params['buy-price']] > dataframe['kama-long'])
                conditions.append(dataframe['close'] < dataframe['kama-short'])
                conditions.append(dataframe['rsi'] < params['rsi-buy-trigger'])
                conditions.append(dataframe['cci'] < params['cci-buy-trigger'])

            if params['buy-method'] == 'trend' or params['buy-method'] == 'both':
                conditions.append(dataframe[params['buy-price']] > dataframe['sar'])



        else:
            conditions.append(dataframe['ohlc4'] > dataframe['sar'])


        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []

        if params['sell-method'] == 'rsi2' or params['sell-method'] == 'both':

            conditions.append(dataframe['close'] > dataframe['kama-short'])
            conditions.append(dataframe['rsi'] > params['rsi-sell-trigger'])
            conditions.append(dataframe['cci'] > params['cci-sell-trigger'])
        if params['sell-method'] == 'trend' or params['sell-method'] == 'both':
            conditions.append(dataframe[params['sell-price']] < dataframe['sar'])


        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
    
    """
    Additional buy/sell timeout override if price drifts
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]

        if current_price > order['price'] * 1.01:
            return True
        return False


    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]

        if current_price < order['price'] * 0.99:
            return True
        return False