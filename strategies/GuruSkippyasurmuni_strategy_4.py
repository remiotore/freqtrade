












'''
Guru Skippyasurmuni - 
Even though Skippy is designed to be easily modified and hyperopted. However, Skippy has alread spent a lot of time thinking about the best way to trade crypto.
He has come to the conclusing - simple is better.  A couple basic indicators and some simple rules will yield the best results in every condition.

'''

'''
docker-compose run --rm freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --strategy GuruSkippyasurmuni -e 2000 --timerange=20220314- --eps
'''

'''
 $1 = 20220314 ; $2 = "day" ;  
 docker-compose run --rm freqtrade backtesting --datadir user_data/data/binanceus --config /freqtrade/user_data/SkippyGod_config.json --export trades  -s SkippyGod --fee 0.00075 --timerange=$1- --breakdown $2 --eps ; 
 docker-compose run --rm freqtrade plot-dataframe -s SkippyGod --config /freqtrade/user_data/SkippyGod_config.json -i 5m --timerange=$1- --indicators2 AROONOSC-5 RSI-5  ; 
 docker-compose run --rm freqtrade plot-profit -s SkippyGod --config /freqtrade/user_data/SkippyGod_config.json -i 5m --timerange=$1-
'''

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date



from random import shuffle
import logging


god_genes = set()



god_genes = {
    'RSI',                  # Relative Strength Index
    'AROONOSC',             # Aroon Oscillator
   }

timeperiods = [5, 6, 12, 15, 50, 55, 100, 110]
operators = [
    "D",  # Disabled gene
    ">",  # Indicator, bigger than cross indicator
    "<",  # Indicator, smaller than cross indicator
    "=",  # Indicator, equal with cross indicator
    "C",  # Indicator, crossed the cross indicator
    "CA",  # Indicator, crossed above the cross indicator
    "CB",  # Indicator, crossed below the cross indicator
    ">R",  # Normalized indicator, bigger than real number
    "=R",  # Normalized indicator, equal with real number
    "<R",  # Normalized indicator, smaller than real number
    "/>R",  # Normalized indicator devided to cross indicator, bigger than real number
    "/=R",  # Normalized indicator devided to cross indicator, equal with real number
    "/<R",  # Normalized indicator devided to cross indicator, smaller than real number
]

TREND_CHECK_CANDLES = 4
DECIMALS = 3



god_genes = list(god_genes)


god_genes_with_timeperiod = list()
for god_gene in god_genes:
    for timeperiod in timeperiods:
        god_genes_with_timeperiod.append(f'{god_gene}-{timeperiod}')



if len(god_genes) == 1:
    god_genes = god_genes*2
if len(timeperiods) == 1:
    timeperiods = timeperiods*2
if len(operators) == 1:
    operators = operators*2


def normalize(df):
    df = (df-df.min())/(df.max()-df.min())
    return df


def gene_calculator(dataframe, indicator):

    if 'CDL' in indicator:
        splited_indicator = indicator.split('-')
        splited_indicator[1] = "0"
        new_indicator = "-".join(splited_indicator)

        indicator = new_indicator

    gene = indicator.split("-")

    gene_name = gene[0]
    gene_len = len(gene)

    if indicator in dataframe.keys():


        return dataframe[indicator]
    else:
        result = None

        if gene_len == 1:

            result = getattr(ta, gene_name)(
                dataframe
            )
            return normalize(result)
        elif gene_len == 2:

            gene_timeperiod = int(gene[1])
            result = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            )
            return normalize(result)

        elif gene_len == 3:

            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            result = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            ).iloc[:, gene_index]
            return normalize(result)

        elif gene_len == 4:

            gene_timeperiod = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            )
            return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))

        elif gene_len == 5:

            gene_timeperiod = int(gene[2])
            gene_index = int(gene[1])
            sharp_indicator = f'{gene_name}-{gene_index}-{gene_timeperiod}'
            dataframe[sharp_indicator] = getattr(ta, gene_name)(
                dataframe,
                timeperiod=gene_timeperiod,
            ).iloc[:, gene_index]
            return normalize(ta.SMA(dataframe[sharp_indicator].fillna(0), TREND_CHECK_CANDLES))


def condition_generator(dataframe, operator, indicator, crossed_indicator, real_num):

    condition = (dataframe['volume'] > 10)


    dataframe[indicator] = gene_calculator(dataframe, indicator)
    dataframe[crossed_indicator] = gene_calculator(dataframe, crossed_indicator)

    indicator_trend_sma = f"{indicator}-SMA-{TREND_CHECK_CANDLES}"
    if operator in ["UT", "DT", "OT", "CUT", "CDT", "COT"]:
        dataframe[indicator_trend_sma] = gene_calculator(dataframe, indicator_trend_sma)

    if operator == ">":
        condition = (
            dataframe[indicator] > dataframe[crossed_indicator]
        )
    elif operator == "=":
        condition = (
            np.isclose(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "<":
        condition = (
            dataframe[indicator] < dataframe[crossed_indicator]
        )
    elif operator == "C":
        condition = (
            (qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])) |
            (qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator]))
        )
    elif operator == "CA":
        condition = (
            qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "CB":
        condition = (
            qtpylib.crossed_below(
                dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == ">R":
        condition = (
            dataframe[indicator] > real_num
        )
    elif operator == "=R":
        condition = (
            np.isclose(dataframe[indicator], real_num)
        )
    elif operator == "<R":
        condition = (
            dataframe[indicator] < real_num
        )
    elif operator == "/>R":
        condition = (
            dataframe[indicator].div(dataframe[crossed_indicator]) > real_num
        )
    elif operator == "/=R":
        condition = (
            np.isclose(dataframe[indicator].div(dataframe[crossed_indicator]), real_num)
        )
    elif operator == "/<R":
        condition = (
            dataframe[indicator].div(dataframe[crossed_indicator]) < real_num
        )
    elif operator == "UT":
        condition = (
            dataframe[indicator] > dataframe[indicator_trend_sma]
        )
    elif operator == "DT":
        condition = (
            dataframe[indicator] < dataframe[indicator_trend_sma]
        )
    elif operator == "OT":
        condition = (

            np.isclose(dataframe[indicator], dataframe[indicator_trend_sma])
        )
    elif operator == "CUT":
        condition = (
            (
                qtpylib.crossed_above(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            ) &
            (
                dataframe[indicator] > dataframe[indicator_trend_sma]
            )
        )
    elif operator == "CDT":
        condition = (
            (
                qtpylib.crossed_below(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            ) &
            (
                dataframe[indicator] < dataframe[indicator_trend_sma]
            )
        )
    elif operator == "COT":
        condition = (
            (
                (
                    qtpylib.crossed_below(
                        dataframe[indicator],
                        dataframe[indicator_trend_sma]
                    )
                ) |
                (
                    qtpylib.crossed_above(
                        dataframe[indicator],
                        dataframe[indicator_trend_sma]
                    )
                )
            ) &
            (
                np.isclose(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            )
        )

    return condition, dataframe

class GuruSkippyasurmuni_strategy_4(IStrategy):
    INTERFACE_VERSION = 2

    DATESTAMP = 0
    COUNT = 0
    custom_info = { }


    buy_params = {
        "buy_crossed_indicator0": "STOCHRSI-0-15",
        "buy_crossed_indicator1": "MACDFIX-0-15",
        "buy_crossed_indicator2": "STOCHRSI-0-55",
        "buy_indicator0": "MACDFIX-0-15",
        "buy_indicator1": "RSI-5",
        "buy_indicator2": "AROONOSC-5",
        "buy_operator0": "D",
        "buy_operator1": "<R",
        "buy_operator2": "=R",
        "buy_real_num0": 0.35,
        "buy_real_num1": 0.10,
        "buy_real_num2": 0,
    }

    sell_params = {
        "sell_crossed_indicator0": "STOCHRSI-0-15",
        "sell_crossed_indicator1": "MACDFIX-0-15",
        "sell_crossed_indicator2": "STOCHRSI-0-15",
        "sell_indicator0": "MACDFIX-0-15",
        "sell_indicator1": "RSI-5",
        "sell_indicator2": "AROONOSC-5",
        "sell_operator0": "D",
        "sell_operator1": ">R",
        "sell_operator2": "=R",
        "sell_real_num0": 0.65,
        "sell_real_num1": 0.90,
        "sell_real_num2": 1.0,
    }

    minimal_roi = {
        "0": 1000.0
    }

    stoploss = -0.35 

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    timeframe = '5m'

    process_only_new_candles = True


    position_adjustment_enable = True
    max_entry_position_adjustment = 4
    dca_multiplier = 0.5

    @property
    def protections(self):
        return [
            {
            "method": "CooldownPeriod",
            "stop_duration_candles": 12
            }
        ]


    buy_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-15", space='buy')
    buy_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='buy')
    buy_crossed_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-55", space='buy')

    buy_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='buy')
    buy_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-5", space='buy')
    buy_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="AROONOSC-5", space='buy')

    buy_operator0 = CategoricalParameter(operators, default="D", space='buy')
    buy_operator1 = CategoricalParameter(operators, default="<R", space='buy')
    buy_operator2 = CategoricalParameter(operators, default="=R", space='buy')

    buy_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS,  default=0.35, space='buy')
    buy_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.30, space='buy')
    buy_real_num2 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.00, space='buy')

    sell_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-15", space='sell')
    sell_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='sell')
    sell_crossed_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="STOCHRSI-0-55", space='sell')

    sell_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default="MACDFIX-0-15", space='sell')
    sell_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default="RSI-5", space='sell')
    sell_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default="AROONOSC-5", space='sell')

    sell_operator0 = CategoricalParameter(operators, default="D", space='sell')
    sell_operator1 = CategoricalParameter(operators, default=">R", space='sell')
    sell_operator2 = CategoricalParameter(operators, default="=R", space='sell')

    sell_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.65, space='sell')
    sell_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.90, space='sell')
    sell_real_num2 = DecimalParameter(0, 1, decimals=DECIMALS, default=0.00, space='sell')



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = metadata['pair']

        if not pair in self.custom_info:

            self.custom_info[pair] = [''] 

        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:


        if sell_reason == 'sell_signal' and trade.calc_profit_ratio(rate) < 0.055:
               return False
        return True
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:

        custom_stake = self.wallets.get_total_stake_amount() / self.config['max_open_trades'] / (self.max_entry_position_adjustment + 1)
        if custom_stake >= min_stake:
            return custom_stake
        elif custom_stake < min_stake:
            return min_stake
        else:
            return proposed_stake


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if(len(dataframe) < 1):
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        if(self.custom_info[trade.pair][self.DATESTAMP] != last_candle['date']):

            self.custom_info[trade.pair][self.DATESTAMP] = last_candle['date']


            if current_profit > -0.065:
                return None

            if last_candle['buy'] > 0:
                filled_buys = trade.select_filled_orders('buy')
                count_of_buys = trade.nr_of_successful_buys
                try:
                    stake_amount = ((count_of_buys * self.dca_multiplier) + 1) * filled_buys[0].cost 
                    if stake_amount < min_stake: 
                        return min_stake
                    else:
                        return stake_amount
                except Exception as exception:
                    return None

        return None
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = list()

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

        buy_indicator = self.buy_indicator2.value
        buy_crossed_indicator = self.buy_crossed_indicator2.value
        buy_operator = self.buy_operator2.value
        buy_real_num = self.buy_real_num2.value
        condition, dataframe = condition_generator(
            dataframe,
            buy_operator,
            buy_indicator,
            buy_crossed_indicator,
            buy_real_num
        )
        conditions.append(condition)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy']=1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = list()

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

        sell_indicator = self.sell_indicator2.value
        sell_crossed_indicator = self.sell_crossed_indicator2.value
        sell_operator = self.sell_operator2.value
        sell_real_num = self.sell_real_num2.value
        condition, dataframe = condition_generator(
            dataframe,
            sell_operator,
            sell_indicator,
            sell_crossed_indicator,
            sell_real_num
        )
        conditions.append(condition)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell']=1

        return dataframe
