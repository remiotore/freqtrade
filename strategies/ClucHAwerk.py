import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class ClucHAwerk(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    buy_params = {
        'bbdelta-close': 0.01021,
        'bbdelta-tail': 0.88118,
        'close-bblower': 0.0022,
        'closedelta-close': 0.00519,
        'rocr-1h': 0.50931,
        'volume': 35
    }

    sell_params = {
        'sell-bbmiddle-close': 1.01283, 
        'sell-rocr-1h': 0.95269
    }

    minimal_roi = {
        "0": 0.11054,
        "2": 0.05569,
        "10": 0.03055,
        "16": 0.02311,
        "82": 0.01267,
        "238": 0.00301,
        "480": 0
    }

    stoploss = -0.02139

    trailing_stop = True
    trailing_stop_positive = 0.09291
    trailing_stop_positive_offset = 0.10651
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """
    
    timeframe = '1m'

    startup_candle_count: int = 168

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        mid, lower = bollinger_bands(dataframe['ha_close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        bollinger = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']

        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)
        
        inf_tf = '1h'
        
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        
        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((      
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail']) &
                    dataframe['ha_close'].lt(dataframe['lower'].shift()) &
                    dataframe['ha_close'].le(dataframe['ha_close'].shift())
            ) |
            (       
                    (dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < params['close-bblower'] * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            dataframe['rocr_1h'].lt(params['sell-rocr-1h']) &
            ((dataframe['ha_close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe

class ClucHAwerk_ETH(ClucHAwerk):

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.02
    ignore_roi_if_buy_signal = True



    buy_params = {
        'bbdelta-close': 0.01566,
        'bbdelta-tail': 0.8478,
        'close-bblower': 0.00998,
        'closedelta-close': 0.00614,
        'rocr-1h': 0.61579,
        'volume': 27
    }



    sell_params = {
        'sell-bbmiddle-close': 1.02894, 
		'sell-rocr-1h': 0.94925
    }



    minimal_roi = {
        "0": 0.14414,
        "13": 0.10123,
        "20": 0.03256,
        "47": 0.0177,
        "132": 0.01016,
        "177": 0.00328,
        "277": 0
    }



    stoploss = -0.02



    trailing_stop = True
    trailing_stop_positive = 0.01018
    trailing_stop_positive_offset = 0.01203
    trailing_only_offset_is_reached = False

class ClucHAwerk_BTC(ClucHAwerk):



    buy_params = {
        'bbdelta-close': 0.01192,
        'bbdelta-tail': 0.96183,
        'close-bblower': 0.01212,
        'closedelta-close': 0.01039,
        'rocr-1h': 0.53422,
        'volume': 27
    }



    sell_params = {
        'sell-bbmiddle-close': 0.98016, 
		'sell-rocr-1h': 0.86823
    }



    minimal_roi = {
        "0": 0.18105,
        "9": 0.10391,
        "49": 0.0447,
        "53": 0.02747,
        "141": 0.01265,
        "312": 0.00499,
        "466": 0
    }



    stoploss = -0.11356



    trailing_stop = True
    trailing_stop_positive = 0.01008
    trailing_stop_positive_offset = 0.04585
    trailing_only_offset_is_reached = False

class ClucHAwerk_USD(ClucHAwerk):



    buy_params = {
        'bbdelta-close': 0.01806,
        'bbdelta-tail': 0.85912,
        'close-bblower': 0.01158,
        'closedelta-close': 0.01466,
        'rocr-1h': 0.51901,
        'volume': 26
    }



    sell_params = {
		'sell-bbmiddle-close': 1.06163, 
		'sell-rocr-1h': 0.63285
    }



    minimal_roi = {
        "0": 0.19315,
        "13": 0.13189,
        "24": 0.08358,
        "103": 0.03894,
        "148": 0.0148,
        "201": 0.00506,
        "447": 0
    }



    stoploss = -0.17725



    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02946
    trailing_only_offset_is_reached = False