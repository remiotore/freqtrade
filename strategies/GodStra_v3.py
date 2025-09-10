# GodStra_v2 Strategy
# Author: @Weather
# GitHub: https://github.com/ReWeatherPort
# Command to optimize:
# freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy roi trailing sell --strategy GodStra_v2
# freqtrade backtesting --config user_data/config.json --strategy  GodStra_v2

from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------

# Define all possible indicators categorized by their types
all_god_genes = {
    'Overlap Studies': {
        'BBANDS-0', 'BBANDS-1', 'BBANDS-2',
        'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA',
        'MA', 'MAMA-0', 'MAMA-1', 'MIDPOINT',
        'MIDPRICE', 'SAR', 'SAREXT', 'SMA',
        'T3', 'TEMA', 'TRIMA', 'WMA',
    },
    'Momentum Indicators': {
        'ADX', 'ADXR', 'APO', 'AROON-0', 'AROON-1',
        'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
        'MACD-0', 'MACD-1', 'MACD-2',
        'MACDEXT-0', 'MACDEXT-1', 'MACDEXT-2',
        'MACDFIX-0', 'MACDFIX-1', 'MACDFIX-2',
        'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM',
        'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC',
        'ROCP', 'ROCR', 'ROCR100', 'RSI',
        'STOCH-0', 'STOCH-1', 'STOCHF-0', 'STOCHF-1',
        'STOCHRSI-0', 'STOCHRSI-1', 'TRIX',
        'ULTOSC', 'WILLR',
    },
    'Volume Indicators': {
        'AD', 'ADOSC', 'OBV',
    },
    'Volatility Indicators': {
        'ATR', 'NATR', 'TRANGE',
    },
    'Price Transform': {
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
    },
    'Cycle Indicators': {
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR-0',
        'HT_PHASOR-1', 'HT_SINE-0', 'HT_SINE-1',
        'HT_TRENDMODE',
    },
    'Pattern Recognition': {
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE',
        'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH',
        'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
        'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
        'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER',
        'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
        'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
        'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
        'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
        'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
        'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
        'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
        'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
        'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
        'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK',
        'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
        'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH',
        'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
        'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
        'CDLXSIDEGAP3METHODS',
    },
    'Statistic Functions': {
        'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE',
        'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
        'STDDEV', 'TSF', 'VAR',
    }
}

# Combine all genes into a single set
god_genes = set().union(*all_god_genes.values())

# Define time periods and operators for conditions
timeperiods = [5, 6, 12, 15, 20, 25, 30, 35]
operators = [
    "D",  # Disabled gene
    ">",  # Indicator > cross indicator
    "<",  # Indicator < cross indicator
    "=",  # Indicator ≈ cross indicator
    "C",  # Indicator crossed cross indicator
    "CA",  # Indicator crossed above cross indicator
    "CB",  # Indicator crossed below cross indicator
    ">R",  # Indicator > real number
    "=R",  # Indicator ≈ real number
    "<R",  # Indicator < real number
    "/>R",  # (Indicator / cross indicator) > real number
    "/=R",  # (Indicator / cross indicator) ≈ real number
    "/<R",  # (Indicator / cross indicator) < real number
    "UT",  # UpTrend status
    "DT",  # DownTrend status
    "OT",  # Off trend status (RANGE)
    "CUT",  # Entered UpTrend status
    "CDT",  # Entered DownTrend status
    "COT"  # Entered Off trend status (RANGE)
]

# Number of candles to check trend status
TREND_CHECK_CANDLES = 4
DECIMALS = 2  # Increased precision for real numbers

god_genes = sorted(god_genes)

# Generate indicators with time periods
god_genes_with_timeperiod = [
    f'{god_gene}-{timeperiod}'
    for god_gene in god_genes
    for timeperiod in timeperiods
]

# Ensure lists have at least two elements for CategoricalParameter
if len(god_genes) == 1:
    god_genes = god_genes * 2
if len(timeperiods) == 1:
    timeperiods = timeperiods * 2
if len(operators) == 1:
    operators = operators * 2

def normalize(series):
    """Normalize the pandas Series."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series.apply(lambda x: 0.5)  # Avoid division by zero
    return (series - min_val) / (max_val - min_val)

def gene_calculator(dataframe, indicator):
    """
    Calculate the specified indicator and normalize it.
    Handles different indicator formats based on the number of components.
    """
    # Handle pattern recognition indicators which don't use time periods
    if 'CDL' in indicator:
        splited_indicator = indicator.split('-')
        splited_indicator[1] = "0"
        new_indicator = "-".join(splited_indicator)
        indicator = new_indicator

    gene = indicator.split("-")
    gene_name = gene[0]
    gene_len = len(gene)

    if indicator in dataframe.columns:
        return dataframe[indicator]
    else:
        try:
            if gene_len == 1:
                # Indicator without time period
                result = getattr(ta, gene_name)(dataframe)
                return normalize(result)
            elif gene_len == 2:
                # Indicator with time period
                gene_timeperiod = int(gene[1])
                result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                return normalize(result)
            elif gene_len == 3:
                # Indicator with multiple time periods (e.g., MACD)
                gene_timeperiod = int(gene[2])
                gene_index = int(gene[1])
                indicator_data = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                if isinstance(indicator_data, pd.DataFrame) and gene_index < indicator_data.shape[1]:
                    result = indicator_data.iloc[:, gene_index]
                else:
                    result = pd.Series(np.nan, index=dataframe.index)
                return normalize(result)
            elif gene_len == 4:
                # Trend operators (e.g., SMA on an indicator)
                gene_timeperiod = int(gene[1])
                sharp_indicator = f'{gene_name}-{gene_timeperiod}'
                dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))
            elif gene_len == 5:
                # Additional trend operators
                gene_timeperiod = int(gene[2])
                gene_index = int(gene[1])
                sharp_indicator = f'{gene_name}-{gene_index}-{gene_timeperiod}'
                indicator_data = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                if isinstance(indicator_data, pd.DataFrame) and gene_index < indicator_data.shape[1]:
                    dataframe[sharp_indicator] = indicator_data.iloc[:, gene_index]
                else:
                    dataframe[sharp_indicator] = np.nan
                return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))
        except Exception as e:
            # Handle any exceptions during indicator calculation
            logger.error(f"Error calculating {indicator}: {e}")
            return np.nan

def condition_generator(dataframe, operator, indicator, crossed_indicator, real_num):
    """
    Generate trading conditions based on the operator and indicators.
    """
    condition = (dataframe['volume'] > 10)  # Basic volume filter

    # Calculate indicators
    dataframe[indicator] = gene_calculator(dataframe, indicator)
    dataframe[crossed_indicator] = gene_calculator(dataframe, crossed_indicator)

    indicator_trend_sma = f"{indicator}-SMA-{TREND_CHECK_CANDLES}"
    if operator in ["UT", "DT", "OT", "CUT", "CDT", "COT"]:
        dataframe[indicator_trend_sma] = gene_calculator(dataframe, indicator_trend_sma)

    # Define conditions based on the operator
    if operator == ">":
        condition &= (dataframe[indicator] > dataframe[crossed_indicator])
    elif operator == "=":
        condition &= (np.isclose(dataframe[indicator], dataframe[crossed_indicator], atol=0.01))
    elif operator == "<":
        condition &= (dataframe[indicator] < dataframe[crossed_indicator])
    elif operator == "C":
        condition &= (
            qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator]) |
            qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "CA":
        condition &= qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == "CB":
        condition &= qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])
    elif operator == ">R":
        condition &= (dataframe[indicator] > real_num)
    elif operator == "=R":
        condition &= (np.isclose(dataframe[indicator], real_num, atol=0.01))
    elif operator == "<R":
        condition &= (dataframe[indicator] < real_num)
    elif operator == "/>R":
        condition &= (dataframe[indicator].div(dataframe[crossed_indicator].replace(0, np.nan)) > real_num)
    elif operator == "/=R":
        condition &= (np.isclose(dataframe[indicator].div(dataframe[crossed_indicator].replace(0, np.nan)), real_num, atol=0.01))
    elif operator == "/<R":
        condition &= (dataframe[indicator].div(dataframe[crossed_indicator].replace(0, np.nan)) < real_num)
    elif operator == "UT":
        condition &= (dataframe[indicator] > dataframe[indicator_trend_sma])
    elif operator == "DT":
        condition &= (dataframe[indicator] < dataframe[indicator_trend_sma])
    elif operator == "OT":
        condition &= (np.isclose(dataframe[indicator], dataframe[indicator_trend_sma], atol=0.01))
    elif operator == "CUT":
        condition &= (
            qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma]) &
            (dataframe[indicator] > dataframe[indicator_trend_sma])
        )
    elif operator == "CDT":
        condition &= (
            qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma]) &
            (dataframe[indicator] < dataframe[indicator_trend_sma])
        )
    elif operator == "COT":
        condition &= (
            (
                qtpylib.crossed_below(dataframe[indicator], dataframe[indicator_trend_sma]) |
                qtpylib.crossed_above(dataframe[indicator], dataframe[indicator_trend_sma])
            ) &
            (np.isclose(dataframe[indicator], dataframe[indicator_trend_sma], atol=0.01))
        )

    return condition, dataframe

class GodStra_v3(IStrategy):
    """
    GodStra_v3 Strategy optimized for higher trading frequency and broader token compatibility.
    """

    INTERFACE_VERSION = 3

    # ROI table optimized for higher frequency
    minimal_roi = {
        "0": 0.293,
        "30": 0.113,
        "76": 0.033,
        "413": 0
    }

    # Stoploss set to -12%
    stoploss = -0.12

    # Shorter timeframe for increased trading frequency
    timeframe = '15m'  # You can change to '15m' for even higher frequency

    # Buy Hyperoptable Parameters/Spaces (increased to two conditions)
    buy_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="EMA-20", space='buy')
    buy_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-14", space='buy')

    buy_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="SMA-100", space='buy')
    buy_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="WILLR-50", space='buy')

    buy_operator0 = CategoricalParameter(operators, default="/<R", space='buy')
    buy_operator1 = CategoricalParameter(operators, default="<R", space='buy')

    buy_real_num0 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.89, space='buy')
    buy_real_num1 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.57, space='buy')

    # Sell Hyperoptable Parameters/Spaces (increased to two conditions)
    sell_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="CDLSHOOTINGSTAR-150", space='sell')
    sell_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="MAMA-1-100", space='sell')

    sell_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="CDLUPSIDEGAP2CROWS-5", space='sell')
    sell_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="CDLHARAMICROSS-150", space='sell')

    sell_operator0 = CategoricalParameter(
        operators, default="<R", space='sell')
    sell_operator1 = CategoricalParameter(
        operators, default="D", space='sell')

    sell_real_num0 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.09, space='sell')
    sell_real_num1 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.81, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators. This strategy calculates indicators dynamically in the entry and exit methods
        to enhance performance and flexibility.
        """
        # Pre-calculate commonly used indicators to avoid redundant calculations
        try:
            # Example: Pre-calculate SMA-30 and RSI-14 if they are used frequently
            indicators_to_precalculate = set([
                "SMA-30", "RSI-14", "EMA-20", "WILLR-50", "CDLSHOOTINGSTAR-150",
                "MAMA-1-100", "CDLUPSIDEGAP2CROWS-5", "CDLHARAMICROSS-150"
            ])
            for indicator in indicators_to_precalculate:
                if indicator not in dataframe.columns:
                    dataframe[indicator] = gene_calculator(dataframe, indicator)
            logger.info("Pre-calculated selected indicators for optimization.")
        except Exception as e:
            logger.error(f"Error in populate_indicators: {e}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy conditions.
        """
        conditions = []

        # First Buy Condition
        buy_indicator = self.buy_indicator0.value
        buy_crossed_indicator = self.buy_crossed_indicator0.value
        buy_operator = self.buy_operator0.value
        buy_real_num = self.buy_real_num0.value
        condition, dataframe = condition_generator(
            dataframe,
            buy_operator,
            buy_indicator,
            buy_crossed_indicator,
            buy_real_num
        )
        conditions.append(condition)

        # Second Buy Condition
        buy_indicator = self.buy_indicator1.value
        buy_crossed_indicator = self.buy_crossed_indicator1.value
        buy_operator = self.buy_operator1.value
        buy_real_num = self.buy_real_num1.value

        condition, dataframe = condition_generator(
            dataframe,
            buy_operator,
            buy_indicator,
            buy_crossed_indicator,
            buy_real_num
        )
        conditions.append(condition)

        # Combine all buy conditions
        if conditions:
            try:
                final_condition = reduce(lambda x, y: x & y, conditions)
                dataframe.loc[
                    final_condition,
                    'enter_long'] = 1
                logger.debug(f"Buy signal generated for {metadata['pair']} at {dataframe.index[-1]}")
            except Exception as e:
                logger.error(f"Error combining buy conditions: {e}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell conditions.
        """
        conditions = []

        # First Sell Condition
        sell_indicator = self.sell_indicator0.value
        sell_crossed_indicator = self.sell_crossed_indicator0.value
        sell_operator = self.sell_operator0.value
        sell_real_num = self.sell_real_num0.value
        condition, dataframe = condition_generator(
            dataframe,
            sell_operator,
            sell_indicator,
            sell_crossed_indicator,
            sell_real_num
        )
        conditions.append(condition)

        # Second Sell Condition
        sell_indicator = self.sell_indicator1.value
        sell_crossed_indicator = self.sell_crossed_indicator1.value
        sell_operator = self.sell_operator1.value
        sell_real_num = self.sell_real_num1.value
        condition, dataframe = condition_generator(
            dataframe,
            sell_operator,
            sell_indicator,
            sell_crossed_indicator,
            sell_real_num
        )
        conditions.append(condition)

        # Combine all sell conditions
        if conditions:
            try:
                final_condition = reduce(lambda x, y: x & y, conditions)
                dataframe.loc[
                    final_condition,
                    'exit_long'] = 1
                logger.debug(f"Sell signal generated for {metadata['pair']} at {dataframe.index[-1]}")
            except Exception as e:
                logger.error(f"Error combining sell conditions: {e}")

        return dataframe