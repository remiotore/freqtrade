import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from freqtrade.exchange import timeframe_to_prev_date
from functools import reduce


###########################################################################################################
##                   BigZ04_TSL3 by Perkmeister, based on BigZ04 by ilya                                 ##
##                                                                                                       ##
##    https://github.com/i1ya/freqtrade-strategies                                                       ##
##    The stratagy most inspired by iterativ (authors of the CombinedBinHAndClucV6)                      ##
##                                                                                                       ##
##    This is a modified version of BigZ04 that uses custom_stoploss() to implement a hard stoploss      ##
##    of 8%, and to replace the roi table with a trailing stoploss to extract more profit when prices    ##
##    start to rise above a profit threshold. It's quite simple and crude and is a 'first stab' at the   ##
##    hard stoploss problem, use live at your own risk ;). The sell signals from SMAOffsetProtectOptV1   ##
##    have been added but are currently disabled as had no benefit.                                      ##
##                                                                                                       ##
###########################################################################################################
##     The main point of this strat is:                                                                  ##
##        -  make drawdown as low as possible                                                            ##
##        -  buy at dip                                                                                  ##
##        -  soft check if market if rising                                                              ##
##        -  hard check is market if fallen                                                              ##
##        -  11 buy signals                                                                              ##
##        -  hard stoploss function preventing from big fall                                             ##
##        -  trailing stoploss while in profit                                                           ##
##        -  no sell signal. Uses custom stoploss                                                        ##
##                                                                                                       ##
###########################################################################################################
##                 GENERAL RECOMMENDATIONS                                                               ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 2 and 4 open trades, with unlimited stake.        ##
##                                                                                                       ##
##   As a pairlist it is recommended to use a static pairlst such as iterativ's orginal:                 ##
##   https://discord.com/channels/700048804539400213/702584639063064586/838038600368783411               ## 
##                                                                                                       ##
##   Ensure that you don't override any variables in your config.json. Especially                        ##
##   the timeframe (must be 5m).                                                                         ##
##                                                                                                       ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS 2 @iterativ (author of the original strategy)                                 ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                     ##
##   ETH: 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                                     ##
##                                                                                                       ##
###########################################################################################################


class BigZ04_TSL3a(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 100.0
    }
    
    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    buy_params = {
        #############
        # Enable/Disable conditions
        "buy_condition_0_enable": True,
        "buy_condition_1_enable": True,
        "buy_condition_2_enable": True,
        "buy_condition_3_enable": True,
        "buy_condition_4_enable": True,
        "buy_condition_5_enable": True,
        "buy_condition_6_enable": False,
        "buy_condition_7_enable": True,
        "buy_condition_8_enable": False,
        "buy_condition_9_enable": True,
        "buy_condition_10_enable": False,
        "buy_condition_11_enable": True,
        "buy_condition_12_enable": False,
        "buy_condition_13_enable": False,
    }

    
    # V1 original
    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 49,
        "high_offset": 1.006,
    }
    
    
    ############################################################################

    # Buy

    buy_condition_0_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_11_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_13_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.7, 1.1, default=0.989, space='buy', optimize=True, load=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.7, 1.1, default=0.982, space='buy', optimize=True, load=True)

    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space='buy', decimals=1, optimize=True, load=True)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=3.8, space='buy', decimals=1, optimize=True, load=True)
    buy_volume_drop_2 = DecimalParameter(1, 10, default=3, space='buy', decimals=1, optimize=True, load=True)
    buy_volume_drop_3 = DecimalParameter(1, 10, default=2.7, space='buy', decimals=1, optimize=True, load=True)

    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=16.5, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=15.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=20.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=35.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_5 = DecimalParameter(10.0, 60.0, default=39.0, space='buy', decimals=1, optimize=True, load=True)

    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=28.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=10.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=14.2, space='buy', decimals=1, optimize=True, load=True)

    buy_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space='buy', decimals=2, optimize=True, load=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space='buy', decimals=2, optimize=True, load=True)

    # hyperopt parameters for custom_stoploss()
    trade_time = IntParameter(25, 65, default=35, space='sell', optimize=False, load=True)
    rsi_1h_val = IntParameter(25, 45, default=32, space='sell', optimize=False, load=True)
    narrow_stop = DecimalParameter(1.005, 1.030, default=1.020, space='sell', decimals=3, optimize=False, load=True)
    wide_stop = DecimalParameter(1.010, 1.045, default=1.035, space='sell', decimals=3, optimize=False, load=True)

    # hyperopt parameters for SMAOffsetProtectOptV1 sell signal
    base_nb_candles_sell = IntParameter(5, 80, default=49, space='sell', optimize=False, load=True)
    high_offset = DecimalParameter(0.99, 1.1, default=1.006, space='sell', optimize=False, load=True)

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        last_close = last_candle['close']
        last_profit = trade.calc_profit_ratio(last_close)
        current_profit = trade.calc_profit_ratio(current_rate)

        buy_tag = 'empty'
        if self.config['runmode'].value in ('live', 'dry_run') and hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        else:
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            buy_signal = dataframe.loc[dataframe['date'] < trade_open_date]
            if not buy_signal.empty:
                buy_signal_candle = buy_signal.iloc[-1]
                buy_tag = buy_signal_candle['buy_tag'] if buy_signal_candle['buy_tag'] != '' else 'empty'
        buy_tags = buy_tag.split()

        if (last_candle['close'] > (last_candle['bb_middleband'] * 1.01)):
            return 'sell signal (' + buy_tag +')'
        if (current_profit <= -0.1):
            return 'stop loss (' + buy_tag +')'
        return None

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:

    #     if (current_profit > 0.3):
    #         return 0.04
    #     elif (current_profit > 0.2):
    #         return 0.03
    #     elif (current_profit > 0.1):
    #         return 0.02
    #     elif (current_profit > 0.05):
    #         return 0.01
    #     elif (current_profit > 0.015):
    #         return 0.0075

    #     return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        # EMA
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        # MACD 
        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # SMA
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ------ ATR stuff
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        buy_12 = (
            self.buy_condition_12_enable.value &

            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_200_1h']) &

            (dataframe['close'] < dataframe['bb_lowerband'] * 0.993) &
            (dataframe['low'] < dataframe['bb_lowerband'] * 0.985) &
            (dataframe['close'].shift() > dataframe['bb_lowerband']) &
            (dataframe['rsi_1h'] < 72.8) &
            (dataframe['open'] > dataframe['close']) &
            
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_12, 'buy_tag'] += '12 '
        conditions.append(buy_12)

        buy_11 = (
            self.buy_condition_11_enable.value &

            (dataframe['close'] > dataframe['ema_200']) &

            (dataframe['hist'] > 0) &
            (dataframe['hist'].shift() > 0) &
            (dataframe['hist'].shift(2) > 0) &
            (dataframe['hist'].shift(3) > 0) &
            (dataframe['hist'].shift(5) > 0) &

            (dataframe['bb_middleband'] - dataframe['bb_middleband'].shift(5) > dataframe['close']/200) &
            (dataframe['bb_middleband'] - dataframe['bb_middleband'].shift(10) > dataframe['close']/100) &
            ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) < (dataframe['close']*0.1)) &
            ((dataframe['open'].shift() - dataframe['close'].shift()) < (dataframe['close'] * 0.018)) &
            (dataframe['rsi'] > 51) &

            (dataframe['open'] < dataframe['close']) &
            (dataframe['open'].shift() > dataframe['close'].shift()) &

            (dataframe['close'] > dataframe['bb_middleband']) &
            (dataframe['close'].shift() < dataframe['bb_middleband'].shift()) &
            (dataframe['low'].shift(2) > dataframe['bb_middleband'].shift(2)) &

            (dataframe['volume'] > 0) # Make sure Volume is not 0
        )
        dataframe.loc[buy_11, 'buy_tag'] += '11 '
        conditions.append(buy_11)

        buy_0 = (
            self.buy_condition_0_enable.value &

            (dataframe['close'] > dataframe['ema_200']) &

            (dataframe['rsi'] < 30) &
            (dataframe['close'] * 1.024 < dataframe['open'].shift(3)) &
            (dataframe['rsi_1h'] < 71) &

            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
        )
        dataframe.loc[buy_0, 'buy_tag'] += '0 '
        conditions.append(buy_0)

        buy_1 = (
            self.buy_condition_1_enable.value &

            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_200_1h']) &

            (dataframe['close'] <  dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_1.value) &
            (dataframe['rsi_1h'] < 69) &
            (dataframe['open'] > dataframe['close']) &
            
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_1, 'buy_tag'] += '1 '
        conditions.append(buy_1)

        buy_2 = (
            self.buy_condition_2_enable.value &

            (dataframe['close'] > dataframe['ema_200']) &

            (dataframe['close'] < dataframe['bb_lowerband'] *  self.buy_bb20_close_bblowerband_safe_2.value) &

            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_2, 'buy_tag'] += '2 '
        conditions.append(buy_2)

        buy_3 = (
            self.buy_condition_3_enable.value &

            (dataframe['close'] > dataframe['ema_200_1h']) &

            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['rsi'] < self.buy_rsi_3.value) &

            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_3.value)) &

            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_3, 'buy_tag'] += '3 '
        conditions.append(buy_3)

        buy_4 = (
            self.buy_condition_4_enable.value &

            (dataframe['rsi_1h'] < self.buy_rsi_1h_1.value) &

            (dataframe['close'] < dataframe['bb_lowerband']) &

            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_4, 'buy_tag'] += '4 '
        conditions.append(buy_4)

        buy_5 = (
            self.buy_condition_5_enable.value &

            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_200_1h']) &

            (dataframe['ema_26'] > dataframe['ema_12']) &
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
            (dataframe['close'] < (dataframe['bb_lowerband'])) &

            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
        )
        dataframe.loc[buy_5, 'buy_tag'] += '5 '
        conditions.append(buy_5)

        buy_6 = (
            self.buy_condition_6_enable.value &

            (dataframe['rsi_1h'] < self.buy_rsi_1h_5.value) &

            (dataframe['ema_26'] > dataframe['ema_12']) &
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
            (dataframe['close'] < (dataframe['bb_lowerband'])) &

            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_6, 'buy_tag'] += '6 '
        conditions.append(buy_6)

        buy_7 = (
            self.buy_condition_7_enable.value &

            (dataframe['rsi_1h'] < self.buy_rsi_1h_2.value) &
            
            (dataframe['ema_26'] > dataframe['ema_12']) &
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
            
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_7, 'buy_tag'] += '7 '
        conditions.append(buy_7)

        buy_8 = (
            self.buy_condition_8_enable.value &

            (dataframe['rsi_1h'] < self.buy_rsi_1h_3.value) &
            (dataframe['rsi'] < self.buy_rsi_1.value) &
            
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &

            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_8, 'buy_tag'] += '8 '
        conditions.append(buy_8)

        buy_9 = (
            self.buy_condition_9_enable.value &

            (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
            (dataframe['rsi'] < self.buy_rsi_2.value) &
            
            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_9, 'buy_tag'] += '9 '
        conditions.append(buy_9)

        buy_10 = (
            self.buy_condition_10_enable.value &

            (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
            (dataframe['close_1h'] < dataframe['bb_lowerband_1h']) &

            (dataframe['hist'] > 0) &
            (dataframe['hist'].shift(2) < 0) &
            (dataframe['rsi'] < 40.5) &
            (dataframe['hist'] > dataframe['close'] * 0.0012) &
            (dataframe['open'] < dataframe['close']) &
            
            (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_10, 'buy_tag'] += '10 '
        conditions.append(buy_10)

        if conditions:
            dataframe.loc[:, 'buy'] = reduce(lambda x, y: x | y, conditions)

        # if conditions:
        #     dataframe.loc[
        #         reduce(lambda x, y: x | y, conditions),
        #         'buy'
        #     ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &                  # Don't be gready, sell fast
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 0
        """
        
        conditions = []
        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=0
        """
        return dataframe
