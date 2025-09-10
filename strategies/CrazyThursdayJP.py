


import datetime
from typing import List, Tuple, Optional
import numpy as np  # noqa
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque
        
class CrazyThursdayJP(IStrategy):


    INTERFACE_VERSION = 3


    minimal_roi = {
        "300" : 0.05,
        "60": 0.1,
        "30": 0.15,
        "0": 0.25,





    }


    stoploss = -0.07


    trailing_stop = False
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.075  # Disabled / not configured
    trailing_only_offset_is_reached = True

    can_short = True

    timeframe = '15m'

    process_only_new_candles = False

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """







        dataframe['rsi'] = ta.RSI(dataframe)

        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']

        dataframe['roc'] = ta.ROC(dataframe)

        dataframe['uo'] = ta.ULTOSC(dataframe)

        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        dataframe['macd'] = ta.MACD(dataframe)['macd']

        dataframe['cci'] = ta.CCI(dataframe)

        dataframe['cmf'] = self.chaikin_money_flow(dataframe, 20)

        dataframe['obv'] = ta.OBV(dataframe)

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['atr'] = qtpylib.atr(dataframe, window=14, exp=False)


        keltner = self.emaKeltner(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_lowerband"] = keltner["lower"]

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bollinger_upperband'] = bollinger['upper']
        dataframe['bollinger_lowerband'] = bollinger['lower']

        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)        
        
        pivots = self.pivot_points(dataframe)
        dataframe['pivot_lows'] = pivots['pivot_lows']
        dataframe['pivot_highs'] = pivots['pivot_highs']





 

        self.initialize_divergences_lists(dataframe)
        self.add_divergences(dataframe, 'rsi')
        self.add_divergences(dataframe, 'stoch')
        self.add_divergences(dataframe, 'roc')
        self.add_divergences(dataframe, 'uo')
        self.add_divergences(dataframe, 'ao')
        self.add_divergences(dataframe, 'macd')
        self.add_divergences(dataframe, 'cci')
        self.add_divergences(dataframe, 'cmf')
        self.add_divergences(dataframe, 'obv')
        self.add_divergences(dataframe, 'mfi')
        self.add_divergences(dataframe, 'adx')

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe[self.resample('total_bullish_divergences')].shift() > 0)









                & self.two_bands_check(dataframe)


                & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe[self.resample('total_bearish_divergences')].shift() > 0)









                & self.two_bands_check(dataframe)


                & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 0
        
        dataframe.loc[
            (
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 0
        return dataframe





























    def resample(self, indicator):

        return indicator

    def two_bands_check(self, dataframe):
        check = (

        ((dataframe[self.resample('low')] < dataframe[self.resample('kc_lowerband')]) & (dataframe[self.resample('high')] > dataframe[self.resample('kc_upperband')])) # 1


        )
        return ~check

    def ema_cross_check(self, dataframe):
        dataframe['ema20_50_cross'] = qtpylib.crossed_below(dataframe[self.resample('ema20')],dataframe[self.resample('ema50')])
        dataframe['ema20_200_cross'] = qtpylib.crossed_below(dataframe[self.resample('ema20')],dataframe[self.resample('ema200')])
        dataframe['ema50_200_cross'] = qtpylib.crossed_below(dataframe[self.resample('ema50')],dataframe[self.resample('ema200')])
        return ~(
            dataframe['ema20_50_cross'] 
            | dataframe['ema20_200_cross'] 
            | dataframe['ema50_200_cross'] 
            )

    def green_candle(self, dataframe):
        return dataframe[self.resample('open')] < dataframe[self.resample('close')]

    def keltner_middleband_check(self, dataframe):
        return (dataframe[self.resample('low')] < dataframe[self.resample('kc_middleband')]) & (dataframe[self.resample('high')] > dataframe[self.resample('kc_middleband')])

    def keltner_lowerband_check(self, dataframe):
        return (dataframe[self.resample('low')] < dataframe[self.resample('kc_lowerband')]) & (dataframe[self.resample('high')] > dataframe[self.resample('kc_lowerband')])

    def bollinger_lowerband_check(self, dataframe):
        return (dataframe[self.resample('low')] < dataframe[self.resample('bollinger_lowerband')]) & (dataframe[self.resample('high')] > dataframe[self.resample('bollinger_lowerband')])

    def bollinger_keltner_check(self, dataframe):
        return (dataframe[self.resample('bollinger_lowerband')] < dataframe[self.resample('kc_lowerband')]) & (dataframe[self.resample('bollinger_upperband')] > dataframe[self.resample('kc_upperband')])

    def ema_check(self, dataframe):
        check = (
            (dataframe[self.resample('ema9')] < dataframe[self.resample('ema20')])
            & (dataframe[self.resample('ema20')] < dataframe[self.resample('ema50')])
            & (dataframe[self.resample('ema50')] < dataframe[self.resample('ema200')]))
        return ~check

    def initialize_divergences_lists(self, dataframe: pd.DataFrame):
        
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''
        
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def add_divergences(self, dataframe: DataFrame, indicator: str):
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(dataframe, indicator)
        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences


        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences



    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str) -> Tuple[pd.Series, pd.Series]:
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            bearish_occurence = self.bearish_divergence_finder(dataframe,
                dataframe[indicator_source],
                high_iterator,
                index)

            if bearish_occurence != None:
                (prev_pivot , current_pivot) = bearish_occurence 
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                length = current_pivot - prev_pivot
                bearish_lines_index = 0
                can_exist = True
                while(True):
                    can_draw = True
                    if bearish_lines_index <= len(bearish_lines):
                        bearish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                    actual_bearish_lines = bearish_lines[bearish_lines_index]
                    for i in range(length + 1):
                        point = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        indicator_point =  bearish_ind_prev_pivot + (bearish_ind_current_pivot - bearish_ind_prev_pivot) * i / length
                        if i != 0 and i != length:
                            if (point <= dataframe['close'][prev_pivot + i] 
                            or indicator_point <= dataframe[indicator_source][prev_pivot + i]):
                                can_exist = False
                        if not np.isnan(actual_bearish_lines[prev_pivot + i]):
                            can_draw = False
                    if not can_exist:
                        break
                    if can_draw:
                        for i in range(length + 1):
                            actual_bearish_lines[prev_pivot + i] = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        break
                    bearish_lines_index = bearish_lines_index + 1
                if can_exist:
                    bearish_divergences[index] = row.close
                    dataframe["total_bearish_divergences"][index] = row.close
                    if index > 30:
                        dataframe["total_bearish_divergences_count"][index-30] = dataframe["total_bearish_divergences_count"][index-30] + 1
                        dataframe["total_bearish_divergences_names"][index-30] = dataframe["total_bearish_divergences_names"][index-30] + indicator_source.upper() + '<br>'

            bullish_occurence = self.bullish_divergence_finder(dataframe,
                dataframe[indicator_source],
                low_iterator,
                index)
            
            if bullish_occurence != None:
                (prev_pivot , current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                length = current_pivot - prev_pivot
                bullish_lines_index = 0
                can_exist = True
                while(True):
                    can_draw = True
                    if bullish_lines_index <= len(bullish_lines):
                        bullish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                    actual_bullish_lines = bullish_lines[bullish_lines_index]
                    for i in range(length + 1):
                        point = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        indicator_point =  bullish_ind_prev_pivot + (bullish_ind_current_pivot - bullish_ind_prev_pivot) * i / length
                        if i != 0 and i != length:
                            if (point >= dataframe['close'][prev_pivot + i] 
                            or indicator_point >= dataframe[indicator_source][prev_pivot + i]):
                                can_exist = False
                        if not np.isnan(actual_bullish_lines[prev_pivot + i]):
                            can_draw = False
                    if not can_exist:
                        break
                    if can_draw:
                        for i in range(length + 1):
                            actual_bullish_lines[prev_pivot + i] = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        break
                    bullish_lines_index = bullish_lines_index + 1
                if can_exist:
                    bullish_divergences[index] = row.close
                    dataframe["total_bullish_divergences"][index] = row.close
                    if index > 30:
                        dataframe["total_bullish_divergences_count"][index-30] = dataframe["total_bullish_divergences_count"][index-30] + 1
                        dataframe["total_bullish_divergences_names"][index-30] = dataframe["total_bullish_divergences_names"][index-30] + indicator_source.upper() + '<br>'
        
        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index):
        if high_iterator[index] == index:
            current_pivot = high_iterator[index]
            occurences = list(dict.fromkeys(high_iterator))
            current_index = occurences.index(high_iterator[index])
            for i in range(current_index-1,current_index-6,-1):            
                prev_pivot = occurences[i]
                if np.isnan(prev_pivot):
                    return
                if ((dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
                or (dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                    return (prev_pivot , current_pivot)
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index):
        if low_iterator[index] == index:
            current_pivot = low_iterator[index]
            occurences = list(dict.fromkeys(low_iterator))
            current_index = occurences.index(low_iterator[index])
            for i in range(current_index-1,current_index-6,-1):
                prev_pivot = occurences[i]
                if np.isnan(prev_pivot):
                    return 
                if ((dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
                or (dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                    return (prev_pivot, current_pivot)
        return None

    def pivot_points(self, dataframe: DataFrame, window: int = 5, pivot_source: int =1) -> DataFrame:
        high_source = None
        low_source = None

        if pivot_source == 1:
            high_source = 'close'
            low_source = 'close'
        elif pivot_source == 0:
            high_source = 'high'
            low_source = 'low'

        pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
        pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
        last_values = deque()

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            last_values.append(row)
            if len(last_values) >= window * 2 + 1:
                current_value = last_values[window]
                is_greater = True
                is_less = True
                for window_index in range(0, window):
                    left = last_values[window_index]
                    right = last_values[2 * window - window_index]
                    local_is_greater, local_is_less = self.check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                    is_greater &= local_is_greater
                    is_less &= local_is_less
                if is_greater:
                    pivot_points_highs[index - window] = getattr(current_value, high_source)
                if is_less:
                    pivot_points_lows[index - window] = getattr(current_value, low_source)
                last_values.popleft()

        if len(last_values) >= window + 2:
            current_value = last_values[-2]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[-2 - window_index - 1]
                right = last_values[-1]
                local_is_greater, local_is_less = self.check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - 1] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - 1] = getattr(current_value, low_source)

        return pd.DataFrame(index=dataframe.index, data={
            'pivot_lows': pivot_points_lows,
            'pivot_highs': pivot_points_highs
        })

    def check_if_pivot_is_greater_or_less(self, current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
        is_greater = True
        is_less = True
        if (getattr(current_value, high_source) < getattr(left, high_source) or
            getattr(current_value, high_source) < getattr(right, high_source)):
            is_greater = False

        if (getattr(current_value, low_source) > getattr(left, low_source) or
            getattr(current_value, low_source) > getattr(right, low_source)):
            is_less = False
        return (is_greater, is_less)

    def emaKeltner(self, dataframe):
        keltner = {}
        atr = qtpylib.atr(dataframe, window=10)
        ema20 = ta.EMA(dataframe, timeperiod=20)
        keltner['upper'] = ema20 + atr
        keltner['mid'] = ema20
        keltner['lower'] = ema20 - atr
        return keltner

    def chaikin_money_flow(self, dataframe, n=20, fillna=False) -> Series:
        """Chaikin Money Flow (CMF)
        It measures the amount of Money Flow Volume over a specific period.
        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
        Args:
            dataframe(pandas.Dataframe): dataframe containing ohlcv
            n(int): n period.
            fillna(bool): if True, fill nan values.
        Returns:
            pandas.Series: New feature generated.
        """
        df = dataframe.copy()
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= df['volume']
        cmf = (mfv.rolling(n, min_periods=0).sum()
            / df['volume'].rolling(n, min_periods=0).sum())
        if fillna:
            cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
        return Series(cmf, name='cmf')
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 5
    