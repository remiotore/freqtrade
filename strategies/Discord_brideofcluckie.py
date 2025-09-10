# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
# --------------------------------
import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame, Series
from technical.indicators import RMI

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']


class brideofcluckie(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.0234,
        'bbdelta-tail': 1.19184,
        'close-bblower': 0.02269,
        'closedelta-close': 0.00235,
        'volume': 78
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.74102
    }

    # ROI table:
    minimal_roi = {
        "0": 0.01,
        "45": 0.0025
    }

    timeframe = '5m'

    stoploss = -0.1

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    informative_timeframe = '1h'

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, dataframe: DataFrame, metadata: dict):

        ssl_down, ssl_up = SSLChannels(dataframe, 25)
        dataframe['ssl_down'] = ssl_down
        dataframe['ssl_up'] = ssl_up
        dataframe['ssl_high'] = (ssl_up > ssl_down).astype('int') * 3
       
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=25)
        
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        dataframe['RMI'] = RMI(dataframe, length=8, mom=4)

        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=3)

        dataframe['go_long'] = (
                (dataframe['ssl_high'] > 0)
                &
                (dataframe['mfi'].shift().rolling(3).mean() > dataframe['mfi'])
                &
                (dataframe['srsi_fk'].shift().rolling(3).mean() > dataframe['srsi_fk'])
                &
                (dataframe['RMI'].shift().rolling(3).mean() < dataframe['RMI'])
                ).astype('int') * 4

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        informative = self.get_informative_indicators(informative.copy(), metadata)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                           ffill=True)
        # don't overwrite the base dataframe's HLCV information
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']


        # strategy BinHV45
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        # strategy ClucMay72018
        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        
        dataframe.loc[
            (dataframe['go_long'] > 0)
            &
            ((  # strategy BinHV45
                    (dataframe['lower'].shift().gt(0)) &
                    (dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
                    (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
                    (dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail'])) &
                    (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                    dataframe['ha_close'].le(dataframe['ha_close'].shift())
            ) |
            (  # strategy ClucMay72018
                    (dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < (params['close-bblower'] * dataframe['bb_lowerband'])) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            ((dataframe['ha_close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1
        return dataframe