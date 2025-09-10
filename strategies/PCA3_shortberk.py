import operator

import numpy as np
from enum import Enum

from pyparsing import Any

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d

import freqtrade.vendor.qtpylib.indicators as qtpylib


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

import pandas as pd
import pandas_ta as pta

pd.options.mode.chained_assignment = None  # default='warn'

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import custom_indicators as cta
from finta import TA as fta

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler
import sklearn.decomposition as skd
from sklearn.svm import SVC

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import random

from prettytable import PrettyTable

"""

PCA - uses Principal Component Analysis to try and reduce the total set of indicators
      to more manageable dimensions, and predict the next gain step.
      
      This works by creating a PCA model of the available technical indicators. This produces a 
      mapping of the indicators and how they affect the outcome (buy/sell/hold). We choose only the
      mappings that have a signficant effect and ignore the others. This significantly reduces the size
      of the problem.
      We then train a classifier model to predict buy or sell signals based on the known outcome in the
      informative data, and use it to predict buy/sell signals based on the real-time dataframe.
      
      Note that this is very slow to start up. This is mostly because we have to build the data on a rolling
      basis to avoid lookahead bias.
      
      In addition to the normal freqtrade packages, these strategies also require the installation of:
        pywavelets
        prettytable
        finta
        --upgrade scikit-learn
        pyparsing

"""

class PCA3_shortberk(IStrategy):


    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.10

    position_adjustment_enable = True
    max_entry_position_adjustment = 3 # initially set to 3 
    max_dca_multiplier = 1.5

    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'

    inf_timeframe = '5m'
    can_short = True
    use_custom_stoploss = True

    use_entry_signal = True
    entry_profit_only = False
    ignore_roi_if_entry_signal = True

    startup_candle_count: int = 128  # must be power of 2
    process_only_new_candles = True
    can_short = True


    inf_mins = timeframe_to_minutes(inf_timeframe)
    data_mins = timeframe_to_minutes(timeframe)
    inf_ratio = int(inf_mins / data_mins)



    lookahead_hours = 1.0
    n_profit_stddevs = 2.0
    n_loss_stddevs = 2.0
    min_f1_score = 0.70

    curr_lookahead = int(12 * lookahead_hours)

    curr_pair = ""
    custom_trade_info = {}


    default_profit_threshold = 0.3
    default_loss_threshold = -0.3
    profit_threshold = default_profit_threshold
    loss_threshold = default_loss_threshold
    dynamic_gain_thresholds = True  # dynamically adjust gain thresholds based on actual mean (beware, training data could be bad)

    dwt_window = startup_candle_count

    num_pairs = 0
    pair_model_info = {}  # holds model-related info for each pair
    classifier_stats = {} # holds statistics for each type of classifier (useful to rank classifiers

    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs

    dbg_scan_classifiers = False  # if True, scan all viable classifiers and choose the best. Very slow!
    dbg_test_classifier = True  # test clasifiers after fitting
    dbg_analyse_pca = False  # analyze PCA weights
    dbg_verbose = False  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    class State(Enum):
        INIT = 1
        POPULATE = 2
        STOPLOSS = 3
        RUNNING = 4




    cexit_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    cexit_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss = DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    @property
    def protections(self):
        return [{'method': 'CooldownPeriod', 'stop_duration_candles': 5}, {'method': 'MaxDrawdown', 'lookback_period_candles': 48, 'trade_limit': 20, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.2}, {'method': 'StoplossGuard', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 2, 'only_per_pair': False}, {'method': 'LowProfitPairs', 'lookback_period_candles': 6, 'trade_limit': 2, 'stop_duration_candles': 60, 'required_profit': 0.02}, {'method': 'LowProfitPairs', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 2, 'required_profit': 0.01}]









    def get_train_buy_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['dwt_maxmin'] >= abs(future_df['loss_threshold'])) &

                    (future_df['dwt_nseq'] < 0) &

                    (future_df['dwt_smooth'] <= future_df['future_min'])
            ), 1.0, 0.0)

        return series

    def get_train_sell_signals(self, future_df: DataFrame):
        series = np.where(
            (
                    (future_df['dwt_maxmin'] >= future_df['profit_threshold']) &

                    (future_df['dwt_nseq'] > 0) &

                    (future_df['dwt_smooth'] >= future_df['future_max'])  # at max of future window
            ), 1.0, 0.0)

        return series



    """
    inf Pair Definitions
    """

    def inf_pairs(self):




        return []


    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:




        curr_pair = metadata['pair']
        self.curr_pair = curr_pair


        self.set_state(curr_pair, self.State.POPULATE)
        self.curr_lookahead = int(12 * self.lookahead_hours)
        self.dbg_curr_df = dataframe

        self.profit_threshold = self.default_profit_threshold
        self.loss_threshold = self.default_loss_threshold

        if PCA3_short.first_time:
            PCA3_short.first_time = False
            print("")
            print("***************************************")
            print("** Warning: startup can be very slow **")
            print("***************************************")

            print("    Lookahead: ", self.curr_lookahead, " candles (", self.lookahead_hours, " hours)")
            print("    Thresholds - Profit:{:.2f}% Loss:{:.2f}%".format(self.profit_threshold,
                                                                        self.loss_threshold))

        print("")
        print(curr_pair)

        if not (curr_pair in self.pair_model_info):
            self.pair_model_info[curr_pair] = {
                'interval': 0,
                'pca_size': 0,
                'pca': None,
                'clf_buy_name': "",
                'clf_buy': None,
                'clf_sell_name': "",
                'clf_sell': None
            }
        else:

            self.pair_model_info[curr_pair]['interval'] = self.pair_model_info[curr_pair]['interval'] - 1

        dataframe = self.add_indicators(dataframe)

        buys, sells = self.create_training_data(dataframe)

        df = dataframe.iloc[:-self.curr_lookahead]
        buys = buys.iloc[:-self.curr_lookahead]
        sells = sells.iloc[:-self.curr_lookahead]


        if self.dbg_verbose:
            print("    training models...")
        self.train_models(curr_pair, df, buys, sells)


        if self.dbg_verbose:
            print("    running predictions...")

        pred_buys = self.predict_buy(dataframe, curr_pair)
        pred_sells = self.predict_sell(dataframe, curr_pair)
        dataframe['predict_buy'] = pred_buys
        dataframe['predict_sell'] = pred_sells

        if self.dbg_verbose:
            print("    updating stoploss data...")
        self.add_stoploss_indicators(dataframe, curr_pair)

        return dataframe





    def add_indicators(self, dataframe: DataFrame) -> DataFrame:

        win_size = max(self.curr_lookahead, 14)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=win_size)
        dataframe['tema_stddev'] = dataframe['tema'].rolling(win_size).std()

        dataframe['emalong'] = ta.EMA(dataframe, timeperiod=win_size)
        dataframe['emashort'] = ta.EMA(dataframe, timeperiod=5)

        period = 14
        smoothD = 3
        SmoothK = 3
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=win_size)
        stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
                dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])

        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=win_size)

        dataframe["dcbb_dist_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_dist_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])

        dataframe['dc_dist'] = (dataframe['dc_upper'] - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]


        dataframe['wr'] = 0.02 * (williams_r(dataframe, period=14) + 50.0)


        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)


        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0



        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)


        dataframe['cti'] = pta.cti(dataframe["close"], length=20)


        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, -1.0))
        dataframe['crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(
            dataframe['close'],
            100)) / 3


        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_480'] = williams_r(dataframe, period=480)


        dataframe['roc_9'] = ta.ROC(dataframe, timeperiod=9)

        dataframe['t3_avg'] = t3_average(dataframe)


        res_series = dataframe['high'].rolling(window=5, center=True).apply(lambda row: is_resistance(row),
                                                                            raw=True).shift(2)
        sup_series = dataframe['low'].rolling(window=5, center=True).apply(lambda row: is_support(row),
                                                                           raw=True).shift(2)
        dataframe['res_level'] = Series(
            np.where(res_series,
                     np.where(dataframe['close'] > dataframe['open'], dataframe['close'], dataframe['open']),
                     float('NaN'))).ffill()
        dataframe['res_hlevel'] = Series(np.where(res_series, dataframe['high'], float('NaN'))).ffill()
        dataframe['sup_level'] = Series(
            np.where(sup_series,
                     np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']),
                     float('NaN'))).ffill()

        dataframe['hl_pct_change_48'] = range_percent_change(dataframe, 'HL', 48)
        dataframe['hl_pct_change_36'] = range_percent_change(dataframe, 'HL', 36)
        dataframe['hl_pct_change_24'] = range_percent_change(dataframe, 'HL', 24)
        dataframe['hl_pct_change_12'] = range_percent_change(dataframe, 'HL', 12)
        dataframe['hl_pct_change_6'] = range_percent_change(dataframe, 'HL', 6)

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)

        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['fast_diff'] = dataframe['fastd'] - dataframe['fastk']

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        dataframe['color'] = np.where((dataframe['close'] > dataframe['open']), 1.0, -1.0)
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['roc_6'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3.0, 1.0, -1.0)
        dataframe['in_the_mood'] = np.where(dataframe['rsi_7'] > dataframe['rsi_7'].rolling(12).mean(), 1.0, -1.0)
        dataframe['moist'] = np.where(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']), 1.0, -1.0)
        dataframe['throbbing'] = np.where(dataframe['roc_6'] > dataframe['roc_6'].rolling(12).mean(), 1.0, -1.0)





















        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_norm'] = self.norm_column(dataframe['mfi'])
        dataframe['mfi_buy'] = np.where((dataframe['mfi_norm'] > 0.5), 1.0, 0.0)
        dataframe['mfi_sell'] = np.where((dataframe['mfi_norm'] <= -0.5), 1.0, 0.0)
        dataframe['mfi_signal'] = dataframe['mfi_buy'] - dataframe['mfi_sell']







        dataframe['atr'] = ta.ATR(dataframe, timeperiod=win_size)





        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']


        dataframe['ewo'] = ewo(dataframe, 50, 200)

        dataframe['uo'] = ta.ULTOSC(dataframe)

        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        dataframe['cci'] = ta.CCI(dataframe)

        self.check_inf(dataframe)

        dataframe.fillna(0.0, inplace=True)

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        proposed_stake = proposed_stake / self.max_dca_multiplier  #  Leaving some reserve incase the market dumps!!!

        return proposed_stake 

    def add_future_data(self, dataframe: DataFrame) -> DataFrame:

        lookahead = self.curr_lookahead

        win_size = max(lookahead, 14)

        future_df = dataframe.copy()


        use_dwt = True
        if use_dwt:
            price_col = 'dwt_full'

            future_df['dwt_full'] = self.get_dwt(future_df['close'])

        else:
            price_col = 'close'
            future_df['dwt_full'] = 0.0


        future_df['dwt_smooth'] = gaussian_filter1d(future_df['dwt_full'], 8)

        future_df['dwt_deriv'] = np.gradient(future_df['dwt_smooth'])
        future_df['dwt_top'] = np.where(qtpylib.crossed_below(future_df['dwt_deriv'], 0.0), 1, 0)
        future_df['dwt_bottom'] = np.where(qtpylib.crossed_above(future_df['dwt_deriv'], 0.0), 1, 0)

        future_df['dwt_diff'] = 100.0 * (future_df['dwt_full'] - future_df['close']) / future_df['close']
        future_df['dwt_smooth_diff'] = 100.0 * (future_df['dwt_full'] - future_df['dwt_smooth']) / future_df['dwt_smooth']

        future_df['dwt_dir'] = 0.0

        future_df['dwt_dir'] = np.where(future_df['dwt_smooth'].diff() >= 0, 1.0, -1.0)

        future_df['dwt_trend'] = np.where(future_df['dwt_dir'].rolling(5).sum() > 0.0, 1.0, -1.0)

        future_df['dwt_gain'] = 100.0 * (future_df['dwt_full'] - future_df['dwt_full'].shift()) / future_df['dwt_full'].shift()

        future_df['dwt_profit'] = future_df['dwt_gain'].clip(lower=0.0)
        future_df['dwt_loss'] = future_df['dwt_gain'].clip(upper=0.0)

        future_df['dwt_mean'] = future_df['dwt_full'].rolling(win_size).mean()
        future_df['dwt_std'] = future_df['dwt_full'].rolling(win_size).std()
        future_df['dwt_profit_mean'] = future_df['dwt_profit'].rolling(win_size).mean()
        future_df['dwt_profit_std'] = future_df['dwt_profit'].rolling(win_size).std()
        future_df['dwt_loss_mean'] = future_df['dwt_loss'].rolling(win_size).mean()
        future_df['dwt_loss_std'] = future_df['dwt_loss'].rolling(win_size).std()

        future_df['dwt_nseq'] = future_df['dwt_dir'].rolling(window=win_size, min_periods=1).sum()

        future_df['dwt_nseq_up'] = future_df['dwt_nseq'].clip(lower=0.0)
        future_df['dwt_nseq_up_mean'] = future_df['dwt_nseq_up'].rolling(window=win_size).mean()
        future_df['dwt_nseq_up_std'] = future_df['dwt_nseq_up'].rolling(window=win_size).std()
        future_df['dwt_nseq_up_thresh'] = future_df['dwt_nseq_up_mean'] + \
                                          self.n_profit_stddevs * future_df['dwt_nseq_up_std']
        future_df['dwt_nseq_sell'] = np.where(future_df['dwt_nseq_up'] > future_df['dwt_nseq_up_thresh'], 1.0, 0.0)

        future_df['dwt_nseq_dn'] = future_df['dwt_nseq'].clip(upper=0.0)
        future_df['dwt_nseq_dn_mean'] = future_df['dwt_nseq_dn'].rolling(window=win_size).mean()
        future_df['dwt_nseq_dn_std'] = future_df['dwt_nseq_dn'].rolling(window=win_size).std()
        future_df['dwt_nseq_dn_thresh'] = future_df['dwt_nseq_dn_mean'] - self.n_loss_stddevs * future_df[
            'dwt_nseq_dn_std']
        future_df['dwt_nseq_buy'] = np.where(future_df['dwt_nseq_dn'] < future_df['dwt_nseq_dn_thresh'], 1.0, 0.0)

        future_df['dwt_recent_min'] = future_df['dwt_smooth'].rolling(window=win_size).min()
        future_df['dwt_recent_max'] = future_df['dwt_smooth'].rolling(window=win_size).max()
        future_df['dwt_maxmin'] = 100.0 * (future_df['dwt_recent_max'] - future_df['dwt_recent_min']) / \
                                  future_df['dwt_recent_max']

        future_df['dwt_low'] = future_df['dwt_smooth'].rolling(window=self.startup_candle_count).min()
        future_df['dwt_high'] = future_df['dwt_smooth'].rolling(window=self.startup_candle_count).max()

        future_df['dwt_at_min'] = np.where(future_df['dwt_smooth'] <= future_df['dwt_recent_min'], 1.0, 0.0)
        future_df['dwt_at_max'] = np.where(future_df['dwt_smooth'] >= future_df['dwt_recent_max'], 1.0, 0.0)
        future_df['dwt_at_low'] = np.where(future_df['dwt_smooth'] <= future_df['dwt_low'], 1.0, 0.0)
        future_df['dwt_at_high'] = np.where(future_df['dwt_smooth'] >= future_df['dwt_high'], 1.0, 0.0)

        future_df['future_close'] = future_df[price_col].shift(-lookahead)

        future_df['future_gain'] = 100.0 * (future_df['future_close'] - future_df[price_col]) / future_df[price_col]
        future_df['future_gain'] = future_df['future_gain'].clip(lower=-5.0, upper=5.0)

        future_df['future_profit'] = future_df['future_gain'].clip(lower=0.0)
        future_df['future_loss'] = future_df['future_gain'].clip(upper=0.0)


        future_df['profit_mean'] = future_df['future_profit'].rolling(win_size).mean()
        future_df['profit_std'] = future_df['future_profit'].rolling(win_size).std()
        future_df['profit_max'] = future_df['future_profit'].rolling(win_size).max()
        future_df['profit_min'] = future_df['future_profit'].rolling(win_size).min()
        future_df['loss_mean'] = future_df['future_loss'].rolling(win_size).mean()
        future_df['loss_std'] = future_df['future_loss'].rolling(win_size).std()
        future_df['loss_max'] = future_df['future_loss'].rolling(win_size).max()
        future_df['loss_min'] = future_df['future_loss'].rolling(win_size).min()



        future_df['profit_threshold'] = future_df['dwt_profit_mean'] + self.n_profit_stddevs * abs(
            future_df['dwt_profit_std'])
        future_df['loss_threshold'] = future_df['dwt_loss_mean'] - self.n_loss_stddevs * abs(future_df['dwt_loss_std'])

        future_df['profit_diff'] = (future_df['future_profit'] - future_df['profit_threshold']) * 10.0
        future_df['loss_diff'] = (future_df['future_loss'] - future_df['loss_threshold']) * 10.0



        future_df['future_dwt'] = future_df['dwt_full'].shift(-lookahead)



        future_df['trend'] = np.where(future_df[price_col] >= future_df[price_col].shift(), 1.0, -1.0)
        future_df['ftrend'] = np.where(future_df['future_close'] >= future_df['future_close'].shift(), 1.0, -1.0)

        future_df['curr_trend'] = np.where(future_df['trend'].rolling(3).sum() > 0.0, 1.0, -1.0)
        future_df['future_trend'] = np.where(future_df['ftrend'].rolling(3).sum() > 0.0, 1.0, -1.0)

        future_df['dwt_dir'] = 0.0
        future_df['dwt_dir'] = np.where(future_df['dwt_full'].diff() >= 0, 1, -1)
        future_df['dwt_dir_up'] = np.where(future_df['dwt_full'].diff() >= 0, 1, 0)
        future_df['dwt_dir_dn'] = np.where(future_df['dwt_full'].diff() < 0, 1, 0)

        future_win = pd.api.indexers.FixedForwardWindowIndexer(window_size=int(win_size))  # don't use a big window

        future_df['future_nseq'] = future_df['curr_trend'].rolling(window=future_win, min_periods=1).sum()






        future_df['future_nseq_up'] = future_df['future_nseq'].clip(lower=0.0)

        future_df['future_nseq_up_mean'] = future_df['future_nseq_up'].rolling(window=future_win).mean()
        future_df['future_nseq_up_std'] = future_df['future_nseq_up'].rolling(window=future_win).std()
        future_df['future_nseq_up_thresh'] = future_df['future_nseq_up_mean'] + self.n_profit_stddevs * future_df[
            'future_nseq_up_std']






        future_df['future_nseq_dn'] = future_df['future_nseq'].clip(upper=0.0)

        future_df['future_nseq_dn_mean'] = future_df['future_nseq_dn'].rolling(future_win).mean()
        future_df['future_nseq_dn_std'] = future_df['future_nseq_dn'].rolling(future_win).std()
        future_df['future_nseq_dn_thresh'] = future_df['future_nseq_dn_mean'] \
                                             - self.n_loss_stddevs * future_df['future_nseq_dn_std']



        future_df['future_min'] = future_df['dwt_smooth'].rolling(window=future_win).min()
        future_df['future_max'] = future_df['dwt_smooth'].rolling(window=future_win).max()

        profit_mean = future_df['future_profit'].mean()
        profit_std = future_df['future_profit'].std()
        loss_mean = future_df['future_loss'].mean()
        loss_std = future_df['future_loss'].std()

        if self.profit_threshold != profit_mean:
            newval = profit_mean + self.n_profit_stddevs * profit_std
            print("    Profit threshold {:.4f} -> {:.4f}".format(self.profit_threshold, newval))
            self.profit_threshold = newval

        if self.loss_threshold != loss_mean:
            newval = loss_mean - self.n_loss_stddevs * abs(loss_std)
            print("    Loss threshold {:.4f} -> {:.4f}".format(self.loss_threshold, newval))
            self.loss_threshold = newval

        return future_df


    def create_training_data(self, dataframe: DataFrame):

        future_df = self.add_future_data(dataframe.copy())

        future_df['train_buy'] = 0.0
        future_df['train_sell'] = 0.0

        future_df['train_buy'] = self.get_train_buy_signals(future_df)
        future_df['train_sell'] = self.get_train_sell_signals(future_df)

        buys = future_df['train_buy'].copy()
        if buys.sum() < 3:
            print("OOPS! <3 ({:.0f}) buy signals generated. Check training criteria".format(buys.sum()))

        sells = future_df['train_sell'].copy()
        if buys.sum() < 3:
            print("OOPS! <3 ({:.0f}) sell signals generated. Check training criteria".format(sells.sum()))

        self.save_debug_data(future_df)
        self.save_debug_indicators(future_df)

        return buys, sells

    def save_debug_data(self, future_df: DataFrame):




        dbg_list = [
            'dwt_full', 'train_buy', 'train_sell',
            'future_gain', 'future_min', 'future_max',
            'profit_min', 'profit_max', 'profit_threshold',
            'loss_min', 'loss_max', 'loss_threshold',
        ]

        if len(dbg_list) > 0:
            for indicator in dbg_list:
                self.add_debug_indicator(future_df, indicator)

        return

    def save_debug_indicators(self, future_df: DataFrame):
        pass
        return


    def add_debug_indicator(self, future_df: DataFrame, indicator):
        dbg_indicator = '%' + indicator
        if not (dbg_indicator in self.dbg_curr_df):
            self.dbg_curr_df[dbg_indicator] = future_df[indicator]


    def roll_smooth(self, col) -> float:


        smooth = gaussian_filter1d(col, 2)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            print("model:", smooth)
            return 0.0

    def get_dwt(self, col):

        a = np.array(col)

        w_mean = a.mean()
        w_std = a.std()
        a_notrend = (a - w_mean) / w_std


        restored_sig = self.dwtModel(a_notrend)

        model = (restored_sig * w_std) + w_mean

        return model

    def roll_get_dwt(self, col) -> float:


        model = self.get_dwt(col)

        length = len(model)
        if length > 0:
            return model[length - 1]
        else:

            return col[len(col)-1]

    def dwtModel(self, data):



        wavelet = 'db8'


        level = 1
        wmode = "smooth"
        tmode = "hard"
        length = len(data)

        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        std = np.std(coeff[level])
        sigma = (1 / 0.6745) * self.madev(coeff[-level])

        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=tmode) for i in coeff[1:])

        model = pywt.waverec(coeff, wavelet, mode=wmode)

        diff = len(model) - len(data)
        return model[0:len(model) - diff]


    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def add_stoploss_indicators(self, dataframe, pair) -> DataFrame:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}
            if not 'had_trend' in self.custom_trade_info[pair]:
                self.custom_trade_info[pair]['had_trend'] = False


        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        dataframe['candle_up'] = np.where(dataframe['close'] >= dataframe['close'].shift(), 1.0, -1.0)
        dataframe['candle_up_trend'] = np.where(dataframe['candle_up'].rolling(5).sum() > 0.0, 1.0, -1.0)
        dataframe['candle_up_seq'] = dataframe['candle_up'].rolling(5).sum()

        dataframe['candle_dn'] = np.where(dataframe['close'] < dataframe['close'].shift(), 1.0, -1.0)
        dataframe['candle_dn_trend'] = np.where(dataframe['candle_up'].rolling(5).sum() > 0.0, 1.0, -1.0)
        dataframe['candle_dn_seq'] = dataframe['candle_up'].rolling(5).sum()

        dataframe['rmi_up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1.0, -1.0)
        dataframe['rmi_up_trend'] = np.where(dataframe['rmi_up'].rolling(5).sum() > 0.0, 1.0, -1.0)

        dataframe['rmi_dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1.0, -1.0)
        dataframe['rmi_dn_count'] = dataframe['rmi_dn'].rolling(8).sum()

        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl_dir'] = 0
        dataframe['ssl_dir'] = np.where(sslup > ssldown, 1.0, -1.0)

        return dataframe

    def norm_column(self, col):
        return self.zscore_column(col)

    def zscore_column(self, col):
        return (col - col.mean()) / col.std()

    def minmax_column(self, col):
        result = col


        if (col.dtype == 'str') or (col.dtype == 'object'):
            result = 0.0
        else:
            result = col
            cmax = max(col)
            cmin = min(col)
            denom = float(cmax - cmin)
            if denom == 0.0:
                result = 0.0
            else:
                result = (col - col.min()) / denom

        return result

    def check_inf(self, dataframe):
        col_name = dataframe.columns.to_series()[np.isinf(dataframe).any()]
        if len(col_name) > 0:
            print("***")
            print("*** Infinity in cols: ", col_name)
            print("***")

    def remove_debug_columns(self, dataframe: DataFrame) -> DataFrame:
        drop_list = dataframe.filter(regex='^%').columns
        if len(drop_list) > 0:
            for col in drop_list:
                dataframe = dataframe.drop(col, axis=1)
            dataframe.reindex()
        return dataframe

    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)

        temp = dataframe.copy()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date']).astype('int64')

        temp = self.remove_debug_columns(temp)

        temp.set_index('date')
        temp.reindex()

        return self.zscore_dataframe(temp).fillna(0.0)

    def zscore_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)
        return ((dataframe - dataframe.mean()) / dataframe.std())

    def scale_dataframe(self, dataframe: DataFrame) -> DataFrame:
        self.check_inf(dataframe)

        temp = dataframe.copy()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date']).astype('int64')

        temp = self.remove_debug_columns(temp)

        temp.reindex()

        scaler = RobustScaler()
        scaler = scaler.fit(temp)
        temp = scaler.transform(temp)
        return temp

    def remove_outliers(self, df_norm: DataFrame, buys, sells):




        df = df_norm.copy()
        df['%temp_buy'] = buys.copy()
        df['%temp_sell'] = sells.copy()

        df2 = df[((df >= -3.0) & (df <= 3.0)).all(axis=1)]

        ndrop = df_norm.shape[0] - df2.shape[0]
        if ndrop > 0:
            b = df2['%temp_buy'].copy()
            s = df2['%temp_sell'].copy()
            df2.drop('%temp_buy', axis=1, inplace=True)
            df2.drop('%temp_sell', axis=1, inplace=True)
            df2.reindex()

            print("    Removed ", ndrop, " outliers")



        else:

            df2 = df_norm
            b = buys
            s = sells
        return df2, b, s

    def build_viable_dataset(self, size: int, df_norm: DataFrame, buys, sells):

        df = df_norm.copy()
        df['%temp_buy'] = buys.copy()
        df['%temp_sell'] = sells.copy()




        df_buy = df.loc[df['%temp_buy'] == 1]
        df_sell = df.loc[df['%temp_sell'] == 1]
        df_nosig = df.loc[(df['%temp_buy'] == 0) & (df['%temp_sell'] == 0)]


        max_signals = int(2*size/3)
        if ((df_buy.shape[0] + df_sell.shape[0]) > max_signals):

            sig_size = int(size/3)
            if (df_buy.shape[0] > sig_size) & (df_sell.shape[0] > sig_size):

                df_buy, _ = train_test_split(df_buy, train_size=sig_size, shuffle=True)
                df_sell, _ = train_test_split(df_sell, train_size=sig_size, shuffle=True)
            else:

                if (df_buy.shape[0] > df_sell.shape[0]):
                    df_buy, _ = train_test_split(df_buy, train_size=max_signals-df_sell.shape[0], shuffle=True)
                else:
                    df_sell, _ = train_test_split(df_sell, train_size=max_signals-df_buy.shape[0], shuffle=True)

        fill_size = size - min(df_buy.shape[0], int(size/3)) - min(df_sell.shape[0], int(size/3))
        df_nosig, _ = train_test_split(df_nosig, train_size=0.6, shuffle=True)


        frames = [df_buy, df_sell, df_nosig]
        df2 = pd.concat(frames)

        df2 = df2.sample(frac=1)

        b = df2['%temp_buy'].copy()
        s = df2['%temp_sell'].copy()
        df2.drop('%temp_buy', axis=1, inplace=True)
        df2.drop('%temp_sell', axis=1, inplace=True)
        df2.reindex()

        return df2, b, s

    def get_binary_labels(self, col):
        binary_encoder = LabelEncoder().fit([min(col), max(col)])
        result = binary_encoder.transform(col)


        return result


    def train_models(self, curr_pair, dataframe: DataFrame, buys, sells):

        count = self.pair_model_info[curr_pair]['interval']
        if (count > 0):
            self.pair_model_info[curr_pair]['interval'] = count - 1
            return
        else:

            self.pair_model_info[curr_pair]['interval'] = random.randint(1, self.curr_lookahead)

        self.pair_model_info[curr_pair]['pca_size'] = 0
        self.pair_model_info[curr_pair]['pca'] = None
        self.pair_model_info[curr_pair]['clf_buy_name'] = ""
        self.pair_model_info[curr_pair]['clf_buy'] = None
        self.pair_model_info[curr_pair]['clf_sell_name'] = ""
        self.pair_model_info[curr_pair]['clf_sell'] = None

        if buys.sum() < 2:
            print("*** ERR: insufficient buys in expected results. Check training data")

            return

        if sells.sum() < 2:
            print("*** ERR: insufficient sells in expected results. Check training data")
            return

        rand_st = 27  # use fixed number for reproducibility

        remove_outliers = False
        if remove_outliers:

            full_df_norm = self.norm_dataframe(dataframe)
            full_df_norm, buys, sells = self.remove_outliers(full_df_norm, buys, sells)
        else:
            full_df_norm = self.norm_dataframe(dataframe).clip(lower=-3.0, upper=3.0)  # supress outliers

        data_size = int(min(975, full_df_norm.shape[0]))

        v_df_norm, v_buys, v_sells = self.build_viable_dataset(data_size, full_df_norm, buys, sells)

        train_size = int(0.6 * data_size)
        test_size = data_size - train_size

        df_train, df_test, train_buys, test_buys, train_sells, test_sells, = train_test_split(v_df_norm,
                                                                                              v_buys,
                                                                                              v_sells,
                                                                                              train_size=0.6,
                                                                                              random_state=rand_st,
                                                                                              shuffle=True)
        if self.dbg_verbose:
            print("     dataframe:", v_df_norm.shape, ' -> train:', df_train.shape, " + test:", df_test.shape)
            print("     buys:", buys.shape, ' -> train:', train_buys.shape, " + test:", test_buys.shape)
            print("     sells:", sells.shape, ' -> train:', train_sells.shape, " + test:", test_sells.shape)

        print("    #training samples:", len(df_train), " #buys:", int(train_buys.sum()), ' #sells:', int(train_sells.sum()))


        buy_labels = self.get_binary_labels(buys)
        sell_labels = self.get_binary_labels(sells)
        train_buy_labels = self.get_binary_labels(train_buys)
        train_sell_labels = self.get_binary_labels(train_sells)
        test_buy_labels = self.get_binary_labels(test_buys)
        test_sell_labels = self.get_binary_labels(test_sells)


        pca = self.get_pca(df_train)

        df_train_pca = DataFrame(pca.transform(df_train))


        print("   ", curr_pair, " - input: ", df_train.shape, " -> pca: ", df_train_pca.shape)

        if df_train_pca.shape[1] <= 1:
            print("***")
            print("** ERR: PCA reduced to 1. Must be training data still in dataframe!")
            print("df_train columns: ", df_train.columns.values)
            print("df_train_pca columns: ", df_train_pca.columns.values)
            print("***")
            return


        buy_ratio = 100.0 * (train_buys.sum() / len(train_buys))
        if (buy_ratio < 0.5):
            print("*** ERR: insufficient number of positive buy labels ({:.2f}%)".format(buy_ratio))
            return

        buy_clf, buy_clf_name = self.get_buy_classifier(df_train_pca, train_buy_labels)

        sell_ratio = 100.0 * (train_sells.sum() / len(train_sells))
        if (sell_ratio < 0.5):
            print("*** ERR: insufficient number of positive sell labels ({:.2f}%)".format(sell_ratio))
            return

        sell_clf, sell_clf_name = self.get_sell_classifier(df_train_pca, train_sell_labels)


        self.pair_model_info[curr_pair]['pca'] = pca
        self.pair_model_info[curr_pair]['pca_size'] = df_train_pca.shape[1]
        self.pair_model_info[curr_pair]['clf_buy_name'] = buy_clf_name
        self.pair_model_info[curr_pair]['clf_buy'] = buy_clf
        self.pair_model_info[curr_pair]['clf_sell_name'] = sell_clf_name
        self.pair_model_info[curr_pair]['clf_sell'] = sell_clf

        if self.dbg_scan_classifiers and self.dbg_verbose:

            df_test_pca = DataFrame(pca.transform(df_test))
            if not (buy_clf is None):
                pred_buys = buy_clf.predict(df_test_pca)
                print("")
                print("Predict - Buy Signals (", type(buy_clf).__name__, ")")
                print(classification_report(test_buy_labels, pred_buys))
                print("")

            if not (sell_clf is None):
                pred_sells = sell_clf.predict(df_test_pca)
                print("")
                print("Predict - Sell Signals (", type(sell_clf).__name__, ")")
                print(classification_report(test_sell_labels, pred_sells))
                print("")

    def get_pca(self, df_norm: DataFrame):

        ncols = df_norm.shape[1]  # allow all components to get the full variance matrix
        whiten = True

        pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)



        ncols = 0
        var_sum = 0.0
        variance_threshold = 0.999

        while ((var_sum < variance_threshold) & (ncols < len(pca.explained_variance_ratio_))):
            var_sum = var_sum + pca.explained_variance_ratio_[ncols]
            ncols = ncols + 1

        if (ncols != df_norm.shape[1]):

            pca = skd.PCA(n_components=ncols, whiten=whiten, svd_solver='full').fit(df_norm)

        self.check_pca(pca, df_norm)

        if self.dbg_analyse_pca and self.dbg_verbose:
            self.analyse_pca(pca, df_norm)

        return pca

    def check_pca(self, pca, df):

        ratios = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        var_big = np.where(ratios >= 0.5)[0]
        if len(var_big) > 0:
            print("    !!! high variance in columns: ", var_big)




        var_0  = np.where(ratios == 0)[0]
        if len(var_0) > 0:
            print("    !!! zero variance in columns: ", var_0)

        inf_rows = loadings[(np.isinf(loadings)).any(axis=1)].index.values.tolist()

        if len(inf_rows) > 0:
            print("    !!! inf values in rows: ", inf_rows)

        na_rows = loadings[loadings.isna().any(axis=1)].index.values.tolist()
        if len(na_rows) > 0:
            print("    !!! na values in rows: ", na_rows)

        zero_rows = loadings[(loadings == 0).any(axis=1)].index.values.tolist()
        if len(zero_rows) > 0:
            print("    !!! zero values in rows (remove indicator?!) : ", zero_rows)

        return


    def analyse_pca(self, pca, df):
        print("")
        print("Variance Ratios:")
        ratios = pca.explained_variance_ratio_
        print(ratios)
        print("")

        loadings = pd.DataFrame(pca.components_.T, index=df.columns.values)

        l2 = loadings.abs()
        l3 = loadings.mul(ratios)
        ranks = loadings.rank()

        loadings['Score'] = l2.sum(axis=1)
        loadings['Score0'] = loadings[loadings.columns.values[0]].abs()
        loadings['Rank'] = loadings['Score'].rank(ascending=False)
        loadings['Rank0'] = loadings['Score0'].rank(ascending=False)
        print("Loadings, by PC0:")
        print(loadings.sort_values('Rank0').head(n=30))
        print("")




        l3a = l3.abs()
        l3['Score'] = l3a.sum(axis=1)
        l3['Rank'] = loadings['Score'].rank(ascending=False)
        print("Loadings, Weighted by Variance Ratio")
        print (l3.sort_values('Rank').head(n=20))

        ranks['Score'] = ranks.sum(axis=1)
        ranks['Rank'] = ranks['Score'].rank(ascending=True)
        print("Rankings per column")
        print(ranks.sort_values('Rank', ascending=True).head(n=30))



    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                                current_rate: float, current_profit: float,
                                min_stake: Optional[float], max_stake: float,
                                current_entry_rate: float, current_exit_rate: float,
                                current_entry_profit: float, current_exit_profit: float,
                                **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:

            return -(trade.stake_amount / 2)

        if current_profit > -0.0125 and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -0.03125 and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -0.09 and trade.nr_of_successful_entries == 3:
            return None

        if current_profit > -0.13 and trade.nr_of_successful_entries == 4:
            return None

        if current_profit > -0.18 and trade.nr_of_successful_entries == 5:
            return None

        if current_profit > -0.0125:

            if trade_duration > 720 and trade.nr_of_successful_entries == 1:
                return None

            if trade_duration > 1440 and trade.nr_of_successful_entries == 2:
                return None

            if trade_duration > 2880 and trade.nr_of_successful_entries == 3:
                return None

            if trade_duration > 5760 and trade.nr_of_successful_entries == 4:
                return None

            if trade_duration > 11520 and trade.nr_of_successful_entries == 5:
                return None


        try:

            stake_amount = filled_entries[0].cost

            if count_of_entries == 1: 
                stake_amount = stake_amount * 1.125
            elif count_of_entries == 2:
                stake_amount = stake_amount * 1.25
            elif count_of_entries == 3:
                stake_amount = stake_amount * 1.375
            elif count_of_entries == 4:
                stake_amount = stake_amount * 1.5
            elif count_of_entries == 5:
                stake_amount = stake_amount * 1.625
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None

    def get_buy_classifier(self, df_norm: DataFrame, results):

        clf = None
        name = ""
        labels = self.get_binary_labels(results)

        if results.sum() <= 2:
            print("***")
            print("*** ERR: insufficient positive results in buy data")
            print("***")
            return clf, name

        if self.pair_model_info[self.curr_pair]['clf_buy']:
            clf = self.pair_model_info[self.curr_pair]['clf_buy']
            clf = clf.fit(df_norm, labels)
            name = self.pair_model_info[self.curr_pair]['clf_buy_name']
        else:
            if self.dbg_scan_classifiers:
                if self.dbg_verbose:
                    print("    Finding best buy classifier:")
                clf, name = self.find_best_classifier(df_norm, labels, tag="buy")
            else:
                clf, name = self.classifier_factory(self.default_classifier, df_norm, labels)
                clf = clf.fit(df_norm, labels)

        return clf, name

    def get_sell_classifier(self, df_norm: DataFrame, results):

        clf = None
        name = ""
        labels = self.get_binary_labels(results)

        if results.sum() <= 2:
            print("***")
            print("*** ERR: insufficient positive results in sell data")
            print("***")
            return clf, name

        if self.pair_model_info[self.curr_pair]['clf_sell']:
            clf = self.pair_model_info[self.curr_pair]['clf_sell']
            clf = clf.fit(df_norm, labels)
            name = self.pair_model_info[self.curr_pair]['clf_sell_name']
        else:
            if self.dbg_scan_classifiers:
                if self.dbg_verbose:
                    print("    Finding best sell classifier:")
                clf, name = self.find_best_classifier(df_norm, labels, tag="sell")
            else:
                clf, name = self.classifier_factory(self.default_classifier, df_norm, labels)
                clf = clf.fit(df_norm, labels)

        return clf, name


    default_classifier = "GradientBoosting"

    classifier_list = [
        'LogisticRegression', 'GaussianNB', 'SGD',
        'GradientBoosting', 'AdaBoost', 'linearSVC', 'sigmoidSVC',
        'LDA'
    ]

    def classifier_factory(self, name, data, labels):
        clf = None

        if name == 'LogisticRegression':
            clf = LogisticRegression(max_iter=10000)
        elif name == 'DecisionTree':
            clf = DecisionTreeClassifier()
        elif name == 'RandomForest':
            clf = RandomForestClassifier()
        elif name == 'GaussianNB':
            clf = GaussianNB()
        elif name == 'MLP':
            param_grid = {
                'hidden_layer_sizes': [(30, 2), (30, 80, 2), (30, 60, 30, 2)],
                'max_iter': [50, 100, 150],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam', 'lbfgs'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
            clf = MLPClassifier(hidden_layer_sizes=(16, 4, 2),
                                max_iter=50,
                                activation='relu',
                                learning_rate='adaptive',
                                alpha=1e-5,
                                solver='lbfgs',
                                verbose=0)


        elif name == 'KNeighbors':
            clf = KNeighborsClassifier(n_neighbors=3)
        elif name == 'SGD':
            clf = SGDClassifier()
        elif name == 'GradientBoosting':
            clf = GradientBoostingClassifier()
        elif name == 'AdaBoost':
            clf = AdaBoostClassifier()
        elif name == 'QDA':
            clf = QuadraticDiscriminantAnalysis()
        elif name == 'linearSVC':
            clf = LinearSVC(dual=False)
        elif name == 'gaussianSVC':
            clf = SVC(kernel='rbf')
        elif name == 'polySVC':
            clf = SVC(kernel='poly')
        elif name == 'sigmoidSVC':
            clf = SVC(kernel='sigmoid')
        elif name == 'Voting':

            c1, _ = self.classifier_factory('AdaBoost', data, labels)
            c2, _ = self.classifier_factory('GaussianNB', data, labels)
            c3, _ = self.classifier_factory('KNeighbors', data, labels)
            c4, _ = self.classifier_factory('DecisionTree', data, labels)
            clf = VotingClassifier(estimators=[('c1', c1), ('c2', c2), ('c3', c3), ('c4', c4)], voting='hard')
        elif name == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif name == 'QDA':
            clf = QuadraticDiscriminantAnalysis()


        else:
            print("Unknown classifier: ", name)
            clf = None
        return clf, name


    def find_best_classifier(self, df, results, tag=""):

        if self.dbg_verbose:
            print("      Evaluating classifiers...")

        scoring = {'accuracy': make_scorer(accuracy_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1_score': make_scorer(f1_score)}

        folds = 5
        clf_dict = {}
        models_scores_table = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1'])

        best_score = -0.1
        best_classifier = ""

        labels = self.get_binary_labels(results)


        df_train, df_test, res_train, res_test = train_test_split(df, labels, train_size=0.8,
                                                                  random_state=27, shuffle=True)




        if res_train.sum() < 2:
            print("    Insufficient +ve (train) results to fit: ", res_train.sum())
            return None, ""

        if res_test.sum() < 2:
            print("    Insufficient +ve (test) results: ", res_test.sum())
            return None, ""

        for cname in self.classifier_list:
            clf, _ = self.classifier_factory(cname, df_train, res_train)

            if clf is not None:

                clf_dict[cname] = clf
                clf = clf.fit(df_train, res_train)

                pred_test = clf.predict(df_test)

                score = f1_score(res_test, pred_test, average='macro')

                if self.dbg_verbose:
                    print("      {0:<20}: {1:.3f}".format(cname, score))

                if score > best_score:
                    best_score = score
                    best_classifier = cname

                if tag:
                    if not (tag in self.classifier_stats):
                        self.classifier_stats[tag] = {}

                    if not (cname in self.classifier_stats[tag]):
                        self.classifier_stats[tag][cname] = { 'count': 0, 'score': 0.0, 'selected': 0}

                    curr_count = self.classifier_stats[tag][cname]['count']
                    curr_score = self.classifier_stats[tag][cname]['score']
                    self.classifier_stats[tag][cname]['count'] = curr_count + 1
                    self.classifier_stats[tag][cname]['score'] = (curr_score * curr_count + score) / (curr_count + 1)

        if best_score <= 0.0:
            print("   No classifier found")
            return None, ""

        clf = clf_dict[best_classifier]

        if best_score < self.min_f1_score:
            print("!!!")
            print("!!! WARNING: F1 score below threshold ({:.3f})".format(best_score))
            print("!!!")
            return None, ""

        if tag:
            if best_classifier in self.classifier_stats[tag]:
                self.classifier_stats[tag][best_classifier]['selected'] = self.classifier_stats[tag][best_classifier] \
                                                                              ['selected'] + 1

        print("       ", tag, " model selected: ", best_classifier, " Score:{:.3f}".format(best_score))


        return clf, best_classifier

    def predict(self, dataframe: DataFrame, pair, clf):

        predict = None

        pca = self.pair_model_info[pair]['pca']

        if clf:

            df_norm = self.norm_dataframe(dataframe)
            df_norm_pca = pca.transform(df_norm)
            predict = clf.predict(df_norm_pca)

        else:
            print("Null CLF for pair: ", pair)

        return predict

    def predict_buy(self, df: DataFrame, pair):
        clf = self.pair_model_info[pair]['clf_buy']

        if clf is None:
            print("    No Buy Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close'].copy()  # just to get the size
            predict = 0.0
            return predict

        predict = self.predict(df, pair, clf)

        return predict

    def predict_sell(self, df: DataFrame, pair):
        clf = self.pair_model_info[pair]['clf_sell']
        if clf is None:
            print("    No Sell Classifier for pair ", pair, " -Skipping predictions")
            self.pair_model_info[pair]['interval'] = min(self.pair_model_info[pair]['interval'], 4)
            predict = df['close']  # just to get the size
            predict = 0.0
            return predict

        predict = self.predict(df, pair, clf)

        return predict



    curr_state = {}

    def set_state(self, pair, state: State):






        self.curr_state[pair] = state

    def get_state(self, pair) -> State:
        return self.curr_state[pair]


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        curr_pair = metadata['pair']
        self.set_state(curr_pair, self.State.RUNNING)
        
        enter_long_conditions = (
            (dataframe['volume'] > 0) &
            (dataframe['mfi'] < 30.0) &
            (dataframe['close'] < dataframe['tema']) &
            qtpylib.crossed_above(dataframe['predict_buy'], 0.5)
        )

        dataframe.loc[enter_long_conditions, "enter_long"] = 1
        dataframe.loc[enter_long_conditions, "enter_tag"] = 'long'

        enter_short_conditions = (
            (dataframe['volume'] > 0) &
            (dataframe['mfi'] > 70.0) &
            (dataframe['close'] > dataframe['tema']) &
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        )

        dataframe.loc[enter_short_conditions, "enter_short"] = 1
        dataframe.loc[enter_short_conditions, "enter_tag"] = 'short'

        return dataframe


    """
    Exit Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        curr_pair = metadata['pair']
        
        exit_long_conditions = [
            dataframe['volume'] > 0,
            dataframe['mfi'] > 70.0,
            dataframe['close'] > dataframe['tema'],
            qtpylib.crossed_above(dataframe['predict_sell'], 0.5)
        ]

        if exit_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"
            ] = 1

        exit_short_conditions = [
            dataframe['volume'] > 0,
            dataframe['mfi'] < 30.0,
            dataframe['close'] < dataframe['tema'],
            qtpylib.crossed_above(dataframe['predict_buy'], 0.5)
        ]

        if exit_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"
            ] = 1

        return dataframe

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:


        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had_trend']

        if current_profit < self.cstop_max_stoploss.value:
            return 0.01

        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':

                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':

                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1


    """
    Custom Exit
    """

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.cexit_pullback_amount.value))
        in_trend = False

        if self.cexit_roi_type.value == 'static':
            min_roi = self.cexit_roi_start.value
        elif self.cexit_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_roi_start.value, self.cexit_roi_end.value, 0,
                                        self.cexit_roi_time.value, trade_dur)
        elif self.cexit_roi_type.value == 'step':
            if trade_dur < self.cexit_roi_time.value:
                min_roi = self.cexit_roi_start.value
            else:
                min_roi = self.cexit_roi_end.value

        if self.cexit_trend_type.value == 'rmi' or self.cexit_trend_type.value == 'any':
            if last_candle['rmi_up_trend'] == 1:
                in_trend = True
        if self.cexit_trend_type.value == 'ssl' or self.cexit_trend_type.value == 'any':
            if last_candle['ssl_dir'] == 1:
                in_trend = True
        if self.cexit_trend_type.value == 'candle' or self.cexit_trend_type.value == 'any':
            if last_candle['candle_up_trend'] == 1:
                in_trend = True

        if in_trend == True and current_profit > 0:

            self.custom_trade_info[trade.pair]['had_trend'] = True

            if self.cexit_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.cexit_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'

            return None

        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had_trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had_trend'] = False
                    return 'trend_roi'
                elif self.cexit_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had_trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None



def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return WR * -100



def t3_average(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe1'] = df['xe1'].fillna(0)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe2'] = df['xe2'].fillna(0)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe3'] = df['xe3'].fillna(0)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe4'] = df['xe4'].fillna(0)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe5'] = df['xe5'].fillna(0)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    df['xe6'] = df['xe6'].fillna(0)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

def is_support(row_data) -> bool:
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) // 2:
            conditions.append(row_data[row] > row_data[row + 1])
        else:
            conditions.append(row_data[row] < row_data[row + 1])
    result = reduce(lambda x, y: x & y, conditions)
    return result

def is_resistance(row_data) -> bool:
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) // 2:
            conditions.append(row_data[row] < row_data[row + 1])
        else:
            conditions.append(row_data[row] > row_data[row + 1])
    result = reduce(lambda x, y: x & y, conditions)
    return result


def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
    """
    Rolling Percentage Change Maximum across interval.

    :param dataframe: DataFrame The original OHLC dataframe
    :param method: High to Low / Open to Close
    :param length: int The length to look back
    """
    if method == 'HL':
        return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe[
            'low'].rolling(length).min()
    elif method == 'OC':
        return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe[
            'close'].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")
