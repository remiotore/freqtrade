# GodStra_v4 Strategy
# Author: @Weather
# GitHub: https://github.com/ReWeatherPort
# Command to dry run:
# freqtrade trade --config user_data/config.json --strategy GodStra_v4 --dry-run
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter
from datetime import datetime, timedelta
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np
import pandas as pd
# Define all possible indicators categorized by their types
all_god_genes = {'Overlap Studies': {'BBANDS-0', 'BBANDS-1', 'BBANDS-2', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MAMA-0', 'MAMA-1', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'}, 'Momentum Indicators': {'ADX', 'ADXR', 'APO', 'AROON-0', 'AROON-1', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX', 'MACD-0', 'MACD-1', 'MACD-2', 'MACDEXT-0', 'MACDEXT-1', 'MACDEXT-2', 'MACDFIX-0', 'MACDFIX-1', 'MACDFIX-2', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH-0', 'STOCH-1', 'STOCHF-0', 'STOCHF-1', 'STOCHRSI-0', 'STOCHRSI-1', 'TRIX', 'ULTOSC', 'WILLR'}, 'Volume Indicators': {'AD', 'ADOSC', 'OBV'}, 'Volatility Indicators': {'ATR', 'NATR', 'TRANGE'}, 'Price Transform': {'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'}, 'Cycle Indicators': {'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR-0', 'HT_PHASOR-1', 'HT_SINE-0', 'HT_SINE-1', 'HT_TRENDMODE'}, 'Pattern Recognition': {'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'}, 'Statistic Functions': {'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'}}
god_genes = set()
for gene_category in all_god_genes.values():
    god_genes |= gene_category
timeperiods = [5, 6, 12, 15, 20, 25, 30, 35, 50]
operators = ['D', '>', '<', '=', 'C', 'CA', 'CB', '>R', '=R', '<R', '/>R', '/=R', '/<R', 'UT', 'DT', 'OT', 'CUT', 'CDT', 'COT']
TREND_CHECK_CANDLES = 4
NORMALIZATION_WINDOW = 200
DECIMALS = 2
god_genes = list(god_genes)
god_genes_with_timeperiod = [f'{god_gene}-{timeperiod}' for god_gene in god_genes for timeperiod in timeperiods]
if len(god_genes) == 1:
    god_genes = god_genes * 2
if len(timeperiods) == 1:
    timeperiods = timeperiods * 2
if len(operators) == 1:
    operators = operators * 2

def rolling_normalize(series, window=NORMALIZATION_WINDOW):
    rolling_min = series.rolling(window=window, min_periods=1).min()
    rolling_max = series.rolling(window=window, min_periods=1).max()
    return (series - rolling_min) / (rolling_max - rolling_min + 1e-10)

def gene_calculator(dataframe, indicator):
    if 'CDL' in indicator:
        splited_indicator = indicator.split('-')
        splited_indicator[1] = '0'
        new_indicator = '-'.join(splited_indicator)
        indicator = new_indicator
    gene = indicator.split('-')
    gene_name = gene[0]
    gene_len = len(gene)
    if indicator in dataframe.columns:
        return dataframe[indicator]
    else:
        result = None
        try:
            if gene_len == 1:
                result = getattr(ta, gene_name)(dataframe)
                return rolling_normalize(result)
            elif gene_len == 2:
                gene_timeperiod = int(gene[1])
                result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                return rolling_normalize(result)
            elif gene_len == 3:
                gene_timeperiod = int(gene[2])
                gene_index = int(gene[1])
                result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index]
                return rolling_normalize(result)
            elif gene_len == 4:
                gene_timeperiod = int(gene[1])
                sharp_indicator = f'{gene_name}-{gene_timeperiod}'
                dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                return rolling_normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))
            elif gene_len == 5:
                gene_timeperiod = int(gene[2])
                gene_index = int(gene[1])
                sharp_indicator = f'{gene_name}-{gene_index}-{gene_timeperiod}'
                dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index]
                return rolling_normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))
        except Exception as e:
            print(f'Error calculating {indicator}: {e}')
            return np.nan

def condition_generator(dataframe, operator, indicator, crossed_indicator, real_num):
    condition = dataframe['volume'] > 10
    dataframe[indicator] = gene_calculator(dataframe, indicator)
    dataframe[crossed_indicator] = gene_calculator(dataframe, crossed_indicator)
    indicator_trend_sma = f'{indicator}-SMA-{TREND_CHECK_CANDLES}'
    if operator in ['UT', 'DT', 'OT', 'CUT', 'CDT', 'COT']:
        dataframe[indicator_trend_sma] = gene_calculator(dataframe, indicator_trend_sma)
    if operator == 'D':
        condition = pd.Series([False] * len(dataframe), index=dataframe.index)
    elif operator == '>':
        condition &= dataframe[indicator] > dataframe[crossed_indicator]
    elif operator == '=':
        condition &= np.isclose(dataframe[indicator], dataframe[crossed_indicator], atol=0.01)
    elif operator == '<':
        condition &= dataframe[indicator] < dataframe[crossed_indicator]
    elif operator == 'C':
        condition &= qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator]) | qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == 'CA':
        condition &= qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == 'CB':
        condition &= qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == '>R':
        condition &= dataframe[indicator] > real_num
    elif operator == '=R':
        condition &= np.isclose(dataframe[indicator], real_num, atol=0.01)
    elif operator == '<R':
        condition &= dataframe[indicator] < real_num
    elif operator == '/>R':
        condition &= dataframe[indicator].div(dataframe[crossed_indicator].replace(0, np.nan)) > real_num
    elif operator == '/=R':
        condition &= np.isclose(dataframe[indicator].div(dataframe[crossed_indicator].replace(0, np.nan)), real_num, atol=0.01)
    elif operator == '/<R':
        condition &= dataframe[indicator].div(dataframe[crossed_indicator].replace(0, np.nan)) < real_num
    elif operator == 'UT':
        condition &= dataframe[indicator] > dataframe[indicator_trend_sma]
    elif operator == 'DT':
        condition &= dataframe[indicator] < dataframe[indicator_trend_sma]
    elif operator == 'OT':
        condition &= np.isclose(dataframe[indicator], dataframe[indicator_trend_sma], atol=0.01)
    elif operator == 'CUT':
        condition &= qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma]) & (dataframe[indicator] > dataframe[indicator_trend_sma])
    elif operator == 'CDT':
        condition &= qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma]) & (dataframe[indicator] < dataframe[indicator_trend_sma])
    elif operator == 'COT':
        condition &= (qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma]) | qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])) & np.isclose(dataframe[indicator], dataframe[indicator_trend_sma], atol=0.01)
    return (condition, dataframe)

class GodStra_v4(IStrategy):
    """
    GodStra_v4 Strategy, optimized for higher profit and shorter holding time on 15m timeframe.
    Added ADX and Bollinger Bands for trend and momentum confirmation.
    Fixed TypeError in BBANDS by using float for nbdevup and nbdevdn.
    """
    INTERFACE_VERSION = 3  # Adjusted for short-term trades
    minimal_roi = {'0': 0.1, '30': 0.05, '60': 0}
    stoploss = -0.08
    timeframe = '15m'
    buy_crossed_indicator0 = CategoricalParameter(god_genes_with_timeperiod, default='ADD-20', space='buy')
    buy_crossed_indicator1 = CategoricalParameter(god_genes_with_timeperiod, default='ASIN-6', space='buy')
    buy_indicator0 = CategoricalParameter(god_genes_with_timeperiod, default='SMA-100', space='buy')
    buy_indicator1 = CategoricalParameter(god_genes_with_timeperiod, default='WILLR-50', space='buy')
    buy_operator0 = CategoricalParameter(operators, default='/<R', space='buy')
    buy_operator1 = CategoricalParameter(operators, default='<R', space='buy')
    buy_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.85, space='buy')
    buy_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.5, space='buy')
    sell_crossed_indicator0 = CategoricalParameter(god_genes_with_timeperiod, default='CDLSHOOTINGSTAR-150', space='sell')
    sell_crossed_indicator1 = CategoricalParameter(god_genes_with_timeperiod, default='MAMA-1-100', space='sell')
    sell_indicator0 = CategoricalParameter(god_genes_with_timeperiod, default='RSI-14', space='sell')
    sell_indicator1 = CategoricalParameter(god_genes_with_timeperiod, default='CDLHARAMICROSS-150', space='sell')
    sell_operator0 = CategoricalParameter(operators, default='D', space='sell')
    sell_operator1 = CategoricalParameter(operators, default='D', space='sell')
    sell_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.9, space='sell')
    sell_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.81, space='sell')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
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
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate trend, momentum, and volatility indicators for 15m timeframe.
        """
        # EMA for trend
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        # RSI for momentum
        dataframe['rsi_14'] = gene_calculator(dataframe, 'RSI-14')
        # MACD for momentum confirmation
        macd = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd_hist'] = macd[2]
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        # Bollinger Bands for breakout and exit (fixed nbdevup and nbdevdn to float)
        bbands = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bbands[0]
        dataframe['bb_middle'] = bbands[1]
        dataframe['bb_lower'] = bbands[2]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy conditions with trend, momentum, and breakout confirmation for 15m timeframe.
        """
        conditions = []
        # Bear market detection: Stop trading if close < EMA-200 or ADX < 25
        bear_condition = (dataframe['close'] > dataframe['ema_200']) & (dataframe['adx'] > 25)
        # Bottom detection: RSI-14 < 0.3 to signal recovery
        bottom_condition = dataframe['rsi_14'] < 0.3
        # Trend and momentum: EMA-12 > EMA-50, RSI-14 > 0.4, MACD Histogram crosses above 0
        trend_condition = dataframe['ema_12'] > dataframe['ema_50']
        momentum_condition = (dataframe['rsi_14'] > 0.4) & qtpylib.crossed_above(dataframe['macd_hist'], 0)
        # Breakout: Close > BB_upper
        breakout_condition = dataframe['close'] > dataframe['bb_upper']
        # Entry logic: Either not in bear market OR recovering from bottom with momentum and breakout
        conditions.append(bear_condition | bottom_condition.shift(1) & trend_condition & momentum_condition & breakout_condition)
        # First Buy Condition
        buy_indicator = self.buy_indicator0.value
        buy_crossed_indicator = self.buy_crossed_indicator0.value
        buy_operator = self.buy_operator0.value
        buy_real_num = self.buy_real_num0.value
        condition, dataframe = condition_generator(dataframe, buy_operator, buy_indicator, buy_crossed_indicator, buy_real_num)
        print(f'Buy Condition 1: {condition.sum()} signals')
        conditions.append(condition)
        # Second Buy Condition
        buy_indicator = self.buy_indicator1.value
        buy_crossed_indicator = self.buy_crossed_indicator1.value
        buy_operator = self.buy_operator1.value
        buy_real_num = self.buy_real_num1.value
        condition, dataframe = condition_generator(dataframe, buy_operator, buy_indicator, buy_crossed_indicator, buy_real_num)
        print(f'Buy Condition 2: {condition.sum()} signals')
        conditions.append(condition)
        # Combine all buy conditions
        if conditions:
            combined_condition = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[combined_condition, 'enter_long'] = 1
            dataframe.loc[combined_condition, 'enter_tag'] = ''
            print(f'Enter Long Signals: {combined_condition.sum()}')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell conditions with faster exits using BB and RSI for 15m timeframe.
        """
        # Exit conditions: (Close < BB_middle AND RSI-14 < 0.5) OR MACD Histogram crosses below 0
        bb_condition = dataframe['close'] < dataframe['bb_middle']
        rsi_condition = dataframe['rsi_14'] < 0.5
        macd_condition = qtpylib.crossed_below(dataframe['macd_hist'], 0)
        # Combine conditions
        exit_condition = bb_condition & rsi_condition | macd_condition
        # Track open positions to avoid excessive exits
        dataframe['position_open'] = 0
        position_open = 0
        for i in range(len(dataframe)):
            if dataframe['enter_long'].iloc[i] == 1:
                position_open = 1
            if position_open == 1 and exit_condition.iloc[i]:
                dataframe.loc[dataframe.index[i], 'exit_long'] = 1
                dataframe.loc[dataframe.index[i], 'exit_reason'] = 'bb_rsi_macd'
                position_open = 0
            dataframe.loc[dataframe.index[i], 'position_open'] = position_open
        print(f'BB Condition (Close < BB_middle): {bb_condition.sum()} signals')
        print(f'RSI Condition (< 0.5): {rsi_condition.sum()} signals')
        print(f'MACD Condition (Cross below 0): {macd_condition.sum()} signals')
        print(f'Combined Exit Signals: {exit_condition.sum()}')
        print(f'Actual Exit Long Signals: {dataframe['exit_long'].sum()}')
        return dataframe