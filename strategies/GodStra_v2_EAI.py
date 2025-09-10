# GodStra_v2_E_Stronger Strategy
# Author: @Weather
# GitHub: https://github.com/ReWeatherPort
# Enhanced version of GodStra_v2_E with further optimizations to improve trading performance

from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import logging

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
god_genes = set()
for gene_category in all_god_genes.values():
    god_genes |= gene_category

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

god_genes = list(god_genes)

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
    return (series - series.min()) / (series.max() - series.min())


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
        result = None
        try:
            if gene_len == 1:
                # Indicator without time period
                result = getattr(ta, gene_name)(dataframe)
                return normalize(result)
            elif gene_len == 2:
                # Indicator with time period
                gene_timeperiod = int(gene[1])
                if gene_name == 'MACD':
                    macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
                    dataframe['MACD_Custom'] = macd['macd']
                    dataframe['MACD_SIGNAL_Custom'] = macd['macdsignal']
                    return normalize(dataframe['MACD_Custom'])
                result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod)
                return normalize(result)
            elif gene_len == 3:
                # Indicator with multiple time periods (e.g., MACD variations)
                gene_timeperiod = int(gene[2])
                gene_index = int(gene[1])
                result = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index]
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
                dataframe[sharp_indicator] = getattr(ta, gene_name)(dataframe, timeperiod=gene_timeperiod).iloc[:, gene_index]
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


class GodStra_v2_EAI(IStrategy):
    """
    GodStra_v2_E_Stronger Strategy with further enhancements to improve trading performance.
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

    # Maximum open trades
    max_open_trades = 5  # Limit the number of simultaneous open trades

    # Cooldown period in minutes
    cooldown_period = 30  # Adjusted to 30 minutes for better trading frequency

    # Trend Filter Parameters
    trend_indicator_short = 'SMA-50'   # Short-term SMA for trend confirmation
    trend_indicator_long = 'SMA-200'   # Long-term SMA for trend confirmation
    trend_operator = ">"  # Price > Both SMAs indicates strong uptrend

    # Buy Hyperoptable Parameters/Spaces (increased to two conditions)
    buy_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="EMA-20", space='buy')  # Changed default for better performance
    buy_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-14", space='buy')  # Added RSI as another indicator

    buy_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACD-0-12", space='buy')  # Changed to MACD component
    buy_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="WILLR-14", space='buy')  # Changed default

    buy_operator0 = CategoricalParameter(operators, default="C", space='buy')  # Changed operator for better signals
    buy_operator1 = CategoricalParameter(operators, default="<R", space='buy')  # Kept for flexibility

    buy_real_num0 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.89, space='buy')
    buy_real_num1 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.30, space='buy')  # Adjusted default for better trade filtering

    # Sell Hyperoptable Parameters/Spaces (increased to two conditions)
    sell_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="CDLSHOOTINGSTAR-150", space='sell')
    sell_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACD-0-26", space='sell')  # Changed to a different MACD component

    sell_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-70", space='sell')  # Added RSI for better sell signals
    sell_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="CDLHARAMICROSS-150", space='sell')

    sell_operator0 = CategoricalParameter(
        operators, default=">R", space='sell')  # Changed operator for sell conditions
    sell_operator1 = CategoricalParameter(
        operators, default="D", space='sell')  # Kept as Disabled for flexibility

    sell_real_num0 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.70, space='sell')  # Adjusted default for RSI sell
    sell_real_num1 = DecimalParameter(
        0, 1, decimals=DECIMALS, default=0.81, space='sell')

    # Initialize last_trade_time within strategy instance
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trade_time = None  # Initialize as None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators. This strategy calculates indicators dynamically in the entry and exit methods
        to enhance performance and flexibility.
        """
        # Calculate trend indicators
        dataframe[self.trend_indicator_short] = gene_calculator(dataframe, self.trend_indicator_short)
        dataframe[self.trend_indicator_long] = gene_calculator(dataframe, self.trend_indicator_long)

        # Calculate additional indicators for buy and sell conditions
        # Calculate RSI and MACD with unique column names to avoid conflicts
        dataframe['RSI_Custom'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['MACD_Custom'] = macd['macd']
        dataframe['MACD_SIGNAL_Custom'] = macd['macdsignal']
        dataframe['MACD_HIST_Custom'] = macd['macdhist']

        # Log the data types of the newly calculated indicators for debugging
        logger.debug(f"RSI_Custom dtype: {dataframe['RSI_Custom'].dtype}")
        logger.debug(f"MACD_Custom dtype: {dataframe['MACD_Custom'].dtype}")
        logger.debug(f"MACD_SIGNAL_Custom dtype: {dataframe['MACD_SIGNAL_Custom'].dtype}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy conditions with additional filters.
        """
        conditions = []

        # Ensure the dataframe index is a DatetimeIndex
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            dataframe.index = pd.to_datetime(dataframe.index)
            logger.debug("Converted dataframe index to DatetimeIndex.")

        # Check trend: Only buy if price is above both SMA-50 and SMA-200
        if self.trend_operator == ">":
            trend_condition = (
                (dataframe['close'] > dataframe[self.trend_indicator_short]) &
                (dataframe['close'] > dataframe[self.trend_indicator_long]) &
                (dataframe['RSI_Custom'] < 70)  # Avoid buying when RSI is high to prevent buying at peaks
            )
        elif self.trend_operator == "<":
            trend_condition = (
                (dataframe['close'] < dataframe[self.trend_indicator_short]) &
                (dataframe['close'] < dataframe[self.trend_indicator_long])
            )
        else:
            trend_condition = True  # Default to True if operator not matched

        conditions.append(trend_condition)

        # Cooldown filter: Only buy if cooldown period has passed
        if self.last_trade_time is not None:
            current_time = dataframe.index[-1]
            try:
                time_since_last_trade = (current_time - self.last_trade_time).total_seconds() / 60
                logger.debug(f"Time since last trade: {time_since_last_trade} minutes.")
                if time_since_last_trade < self.cooldown_period:
                    logger.debug("In cooldown period. Skipping buy conditions.")
                    return dataframe  # Skip buy conditions if in cooldown
            except Exception as e:
                logger.error(f"Error calculating cooldown period: {e}")
                # Proceed without applying cooldown if error occurs

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

        # Add Price Drop Confirmation to avoid buying before the drop has finished
        dataframe['Price_Drop'] = (dataframe['close'].rolling(window=20).max() - dataframe['close']) / dataframe['close'].rolling(window=20).max()
        drop_condition = dataframe['Price_Drop'] > 0.05  # Ensure at least a 5% drop has occurred
        conditions.append(drop_condition)

        # Combine all buy conditions
        if conditions:
            final_condition = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[
                final_condition,
                'enter_long'] = 1
            # Update last_trade_time if a buy signal is generated
            if final_condition.iloc[-1]:
                self.last_trade_time = dataframe.index[-1]
                logger.debug(f"Buy signal triggered at {self.last_trade_time}.")

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

        # Momentum Confirmation: Avoid selling during momentum reversals
        # Ensure that MACD_Custom and MACD_SIGNAL_Custom are numeric
        dataframe['MACD_Custom'] = pd.to_numeric(dataframe['MACD_Custom'], errors='coerce')
        dataframe['MACD_SIGNAL_Custom'] = pd.to_numeric(dataframe['MACD_SIGNAL_Custom'], errors='coerce')
        dataframe['MACD_Diff'] = dataframe['MACD_Custom'] - dataframe['MACD_SIGNAL_Custom']
        momentum_condition = dataframe['MACD_Diff'] < 0  # MACD line crosses below the signal line
        conditions.append(momentum_condition)

        # Combine all sell conditions
        if conditions:
            final_condition = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[
                final_condition,
                'exit_long'] = 1

        return dataframe