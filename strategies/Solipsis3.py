import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from cachetools import TTLCache

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import custom_indicators as cta

"""
Solipsis - By @werkkrew and @JimmyNixx
This strategy is an evolution of our previous framework "Schism" which can be found in this repository.  While Schism has been superceded by this
strategy there may still be valuable examples and ideas in it.

We ask for nothing in return except that if you make changes which bring you greater success than what has been provided, you share those ideas back to us
and the rest of the community. Also, please don't nag us with a million questions and especially don't blame us if you lose a ton of money using this.

We take no responsibility for any success or failure you have using this strategy.

Apes together strong.
This is not financial advice.
We like the stock.
Where lambo?

*************
This is a very advanced strategy.  It requires a lot of configuration, optimization, and understanding of how it works and what it does before use.
It **will not** work at all for you "out of the box".  If you download it and run an immediate backtest the odds are the results will be awful.

Please review the code, understand it, and attempt to do as much due diligence as you can before asking questions about it.
*************

FEATURES:
    - Dynamic ROI
        - Several options, initial idea was to ride trends past ROI in a similar way to trailing stoploss but using indicators.
        - Fallback choices includes table, roc, atr, and others.  Has the ability to set ROI table values dynamically based on indicator math.
    - Custom Stoploss
        - Generally a vanilla implementation of Freqtrade custom stoploss but tries to do some clever things.  Uses indicator data. (Thanks @JoeSchr!)
    - Dynamic informative indicators based on certain stake currences and whitelist contents.
        - If BTC/STAKE is not in whitelist, make sure to use that for an informative.
        - If your stake is BTC or ETH, use COIN/FIAT and BTC/FIAT as informatives.
    - Ability to provide custom parameters on a per-pair or group of pairs basis, this includes buy/sell/minimal_roi/dynamic_roi/custom_stop settings, if one desired.
    - Custom indicator file to keep primary strategy clean(ish).
        - Most (but not all) of what is in there is taken from freqtrade/technical with some slight modification, removes dependenacy on that import and allows
          for some customization without having to edit those files directly.
    - Child strategy for stake specific settings and different settings for different instances, hoping to keep this strategy file relatively
      clutter-free from the extensive options especially when using per-pair settings.

STRATEGY NOTES:
    - If trading on a stablecoin or fiat stake (such as USD, EUR, USDT, etc.) is *highly recommended* that you remove BTC/STAKE
      from your whitelist as this strategy performs much better on alts when using BTC as an informative but does not buy any BTC
      itself.
    - It is recommended to configure protections *if/as* you will use them in live and run *some* hyperopt/backtest with
      "--enable-protections" as this strategy will hit a lot of stoplosses so the stoploss protection is helpful
      to test. *However* - this option makes hyperopt very slow, so run your initial backtest/hyperopts without this
      option. Once you settle on a baseline set of options, do some final optimizations with protections on.
    - It is *not* recommended to use freqtrades built-in trailing stop, nor to hyperopt for that.
    - It is *highly* recommended to hyperopt this with '--spaces buy' only and at least 1000 total epochs several times. There are
      a lot of variables being hyperopted and it may take a lot of epochs to find the right settings.
    - It is possible to hyperopt the custom stoploss and dynamic ROI settings, however a change to the freqtrade code is needed.  I have done
      this in a fork on github and I use it personally, but this code will likely never get merged upstream so use with extreme caution.
      (https://github.com/werkkrew/freqtrade/tree/hyperopt)
    - Hyperopt Notes:
        - Hyperopting buy/custom-stoploss/dynamic-roi together takes a LOT of repeat 1000 epoch runs to get optimal results.  There
          are a ton of variables moving around and often times the reported best epoch is not desirable.
        - Avoid hyperopt results with small avg. profit and avg. duration of < 60m (in my opinion.)
        - I find the best results come from SharpeHyperOptLoss
        - I personally re-run it until I find epochs with at least 0.5% avg profit and a 10:1 w/l ratio as my personal preference.
    - It is *recommended* to leave this file untouched and do your configuration / optimizations from the child strategy Solipsis.py.

    - Example of unique buy/sell params per pair/group of pairs:

    custom_pair_params = [
        {
            'pairs': ('ABC/XYZ', 'DEF/XYZ'),
            'buy_params': {},
            'sell_params': {},
            'minimal_roi': {}
        }
    ]

TODO:
    - Continue to hunt for a better all around buy signal.
    - Tweak ROI Trend Ride
        - Adjust pullback to be more dynamic, seems to get out a tad bit early in many cases.
        - Consider a way to identify very large/fast spikes when RMI has not yet reacted to stay in past ROI point.
    - Further enchance and optimize custom stop loss
        - Continue to evaluate good circumstances to bail and sell vs hold on for recovery
        - Curent implementation seems to work pretty well but feel like there is room for improvement.
    - Develop a PR to fully support hyperopting the custom_stoploss and dynamic_roi spaces?
"""

class Solipsis3(IStrategy):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'base-mp': 30,
        'base-rmi-fast': 50,
        'base-rmi-slow': 30,
        'inf-guard': 'upper',
        'inf-pct-adr-bot': 0.10,
        'inf-pct-adr-top': 0.85,
        'xbtc-base-rmi': 50,
        'xbtc-inf-rmi': 20,
        'xtra-base-fiat-rmi': 45,
        'xtra-base-stake-rmi': 69,
        'xtra-inf-stake-rmi': 27
    }

    sell_params = {}

    custom_pair_params = []

    minimal_roi = {
        "0": 0.01,
        "360": 0.005,
        "720": 0
    }

    dynamic_roi = {
        'enabled': True,          # enable dynamic roi which uses trennds and indicators to dynamically manipulate the roi table
        'profit-factor': 400,     # factor for forumla of how far below peak profit to trigger sell
        'rmi-start': 30,          # starting value for rmi-slow to be considered a positive trend
        'rmi-end': 70,            # ending value
        'grow-delay': 0,          # delay on growth
        'grow-time': 720,         # finish time of growth
        'fallback': 'table',      # if no trend, do what? (table, roc, atr, roc-table, atr-table)
        'min-roc-atr': 0.0075     # minimum roi value to return in roc or atr mode
    }

    use_custom_stoploss = True
    custom_stop = {

        'decay-time': 1080,       # minutes to reach end, I find it works well to match this to the final ROI value
        'decay-delay': 0,         # minutes to wait before decay starts
        'decay-start': -0.30,     # starting value: should be the same or smaller than initial stoploss
        'decay-end': -0.03,       # ending value

        'cur-min-diff': 0.03,     # diff between current and minimum profit to move stoploss up to min profit point
        'cur-threshold': -0.02,   # how far negative should current profit be before we consider moving it up based on cur/min or roc
        'roc-bail': -0.03,        # value for roc to use for dynamic bailout
        'rmi-trend': 50,          # rmi-slow value to pause stoploss decay
        'bail-how': 'immediate',  # set the stoploss to the atr offset below current price, or immediate

        'pos-trail': False,       # enable trailing once positive
        'pos-threshold': 0.005,   # trail after how far positive
        'pos-trail-dist': 0.015   # how far behind to place the trail
    }

    stoploss = custom_stop['decay-start']

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 72
    process_only_new_candles = False

    custom_trade_info = {}
    custom_fiat = "USD" # Only relevant if stake is BTC or ETH
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes

    """
    Informative Pair Definitions
    """
    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]

        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]
            informative_pairs += [(stake_fiat, self.inf_timeframe)]

        else:
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                informative_pairs += [(btc_stake, self.timeframe)]
                informative_pairs += [(btc_stake, self.inf_timeframe)]

        return informative_pairs

    """
    Indicator Definitions
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])

        dataframe['rmi-slow'] = cta.RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = cta.RMI(dataframe, length=8, mom=4)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=24)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)

        dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['mp']  = ta.RSI(dataframe['roc-mp'], timeperiod=6)

        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        informative['1d_high'] = informative['close'].rolling(24).max()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d_high'] - informative['3d_low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)


        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rmi"] = cta.RMI(coin_fiat_tf, length=21, mom=5)

            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            dataframe[f"{stake}_rmi"] = cta.RMI(stake_fiat_tf, length=21, mom=5)

            stake_fiat_inf_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.inf_timeframe)
            stake_fiat_inf_tf[f"{stake}_rmi"] = cta.RMI(stake_fiat_inf_tf, length=48, mom=5)
            dataframe = merge_informative_pair(dataframe, stake_fiat_inf_tf, self.timeframe, self.inf_timeframe, ffill=True)

        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:

                btc_stake_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
                dataframe['BTC_rmi'] = cta.RMI(btc_stake_tf, length=14, mom=3)

                btc_stake_inf_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.inf_timeframe)
                btc_stake_inf_tf['BTC_rmi'] = cta.RMI(btc_stake_inf_tf, length=48, mom=5)
                dataframe = merge_informative_pair(dataframe, btc_stake_inf_tf, self.timeframe, self.inf_timeframe, ffill=True)

        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['roc'] = dataframe[['date', 'roc']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr'] = dataframe[['date', 'atr']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-slow'] = dataframe[['date', 'rmi-slow']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend'] = dataframe[['date', 'rmi-up-trend']].copy().set_index('date')

        return dataframe

    """
    Buy Signal
    """
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'buy')
        conditions = []

        if params['inf-guard'] == 'upper' or params['inf-guard'] == 'both':
            conditions.append(
                (dataframe['close'] <= dataframe[f"3d_low_{self.inf_timeframe}"] +
                (params['inf-pct-adr-top'] * dataframe[f"adr_{self.inf_timeframe}"]))
            )

        if params['inf-guard'] == 'lower' or params['inf-guard'] == 'both':
            conditions.append(
                (dataframe['close'] >= dataframe[f"3d_low_{self.inf_timeframe}"] +
                (params['inf-pct-adr-bot'] * dataframe[f"adr_{self.inf_timeframe}"]))
            )

        conditions.append(
            (dataframe['rmi-dn-trend'] == 1) &
            (dataframe['rmi-slow'] >= params['base-rmi-slow']) &
            (dataframe['rmi-fast'] <= params['base-rmi-fast']) &
            (dataframe['mp'] <= params['base-mp'])
        )

        if self.config['stake_currency'] in ('BTC', 'ETH'):
            conditions.append(
                (dataframe[f"{self.config['stake_currency']}_rmi"] < params['xtra-base-stake-rmi']) |
                (dataframe[f"{self.custom_fiat}_rmi"] > params['xtra-base-fiat-rmi'])
            )
            conditions.append(dataframe[f"{self.config['stake_currency']}_rmi_{self.inf_timeframe}"] < params['xtra-inf-stake-rmi'])

        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                conditions.append(
                    (dataframe['BTC_rmi'] < params['xbtc-base-rmi']) &
                    (dataframe[f"BTC_rmi_{self.inf_timeframe}"] > params['xbtc-inf-rmi'])
                )

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    """
    Sell Signal
    """
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['sell'] = 0

        return dataframe

    """
    Custom Stoploss
    """
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        params = self.get_pair_params(pair, 'custom_stop')

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        min_profit = trade.calc_profit_ratio(trade.min_rate)
        max_profit = trade.calc_profit_ratio(trade.max_rate)
        profit_diff = current_profit - min_profit

        decay_stoploss = cta.linear_growth(params['decay-start'], params['decay-end'], params['decay-delay'], params['decay-time'], trade_dur)

        if params['pos-trail'] == True:
            if current_profit > params['pos-threshold']:
                return current_profit - params['pos-trail-dist']

        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            roc = dataframe['roc'].iat[-1]
            atr = dataframe['atr'].iat[-1]
            rmi_slow = dataframe['rmi-slow'].iat[-1]

        else:
            roc = self.custom_trade_info[trade.pair]['roc'].loc[current_time]['roc']
            atr = self.custom_trade_info[trade.pair]['atr'].loc[current_time]['atr']
            rmi_slow = self.custom_trade_info[trade.pair]['rmi-slow'].loc[current_time]['rmi-slow']

        if current_profit < params['cur-threshold']:

            if (roc/100) <= params['roc-bail']:
                if params['bail-how'] == 'atr':
                    return ((current_rate - atr)/current_rate) - 1
                elif params['bail-how'] == 'immediate':
                    return current_rate
                else:
                    return decay_stoploss

        if (current_profit > min_profit) or roc > 0 or rmi_slow >= params['rmi-trend']:
            if profit_diff > params['cur-min-diff'] and current_profit < 0:
                return min_profit
            return -1

        return decay_stoploss

    """
    Freqtrade ROI Overload for dynamic ROI functionality
    """
    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        dynamic_roi = self.get_pair_params(trade.pair, 'dynamic_roi')
        minimal_roi = self.get_pair_params(trade.pair, 'minimal_roi')

        if not dynamic_roi or not minimal_roi:
            return None, None

        _, table_roi = self.min_roi_reached_entry(trade_dur, trade.pair)

        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                roc = dataframe['roc'].iat[-1]
                atr = dataframe['atr'].iat[-1]
                rmi_slow = dataframe['rmi-slow'].iat[-1]
                rmi_trend = dataframe['rmi-up-trend'].iat[-1]

            else:
                roc = self.custom_trade_info[trade.pair]['roc'].loc[current_time]['roc']
                atr = self.custom_trade_info[trade.pair]['atr'].loc[current_time]['atr']
                rmi_slow = self.custom_trade_info[trade.pair]['rmi-slow'].loc[current_time]['rmi-slow']
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend'].loc[current_time]['rmi-up-trend']

            d = dynamic_roi
            open_rate = trade.open_rate
            max_profit = trade.calc_profit_ratio(trade.max_rate)

            atr_roi = max(d['min-roc-atr'], ((open_rate + atr) / open_rate) - 1)
            roc_roi = max(d['min-roc-atr'], (roc/100))

            if d['fallback'] == 'atr':
                min_roi = atr_roi

            elif d['fallback'] == 'roc':
                min_roi = roc_roi

            elif d['fallback'] == 'atr-table':
                min_roi = max(table_roi, atr_roi)

            elif d['fallback'] == 'roc-table':
                min_roi = max(table_roi, roc_roi)

            else:
                min_roi = table_roi

            rmi_grow = cta.linear_growth(d['rmi-start'], d['rmi-end'], d['grow-delay'], d['grow-time'], trade_dur)
            profit_factor = (1 - (rmi_slow / d['profit-factor']))
            pullback_buffer = (max_profit * profit_factor)

            if (rmi_trend == 1) and (rmi_slow > rmi_grow):
                if (current_profit < pullback_buffer) and max_profit > min_roi:

                    min_roi = current_profit / 2
                else:
                    min_roi = 100

        else:
            min_roi = table_roi

        return trade_dur, min_roi

    def min_roi_reached_entry(self, trade_dur: int, pair: str = 'backtest') -> Tuple[Optional[int], Optional[float]]:
        minimal_roi = self.get_pair_params(pair, 'minimal_roi')

        roi_list = list(filter(lambda x: x <= trade_dur, minimal_roi.keys()))
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)
        min_roi = minimal_roi[roi_entry]

        return roi_entry, min_roi

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.dynamic_roi and 'enabled' in self.dynamic_roi and self.dynamic_roi['enabled']:
            _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur, trade.pair)
        if roi is None:
            return False
        else:
            return current_profit > roi

    """
    Trade Timeout Overloads
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        if current_price > order['price'] * 1.01:
            return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ask_strategy = self.config.get('ask_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{ask_strategy['price_side']}s"][0][0]
        if current_price < order['price'] * 0.99:
            return True
        return False

    """
    Custom Methods
    """


    def get_pair_params(self, pair: str, params: str) -> Dict:
        buy_params, sell_params = self.buy_params, self.sell_params
        minimal_roi, dynamic_roi = self.minimal_roi, self.dynamic_roi
        custom_stop = self.custom_stop

        if self.custom_pair_params:

            for item in self.custom_pair_params:
                if 'pairs' in item and pair in item['pairs']:
                    custom_params = item
                    if 'buy_params' in custom_params:
                        buy_params = custom_params['buy_params']
                    if 'sell_params' in custom_params:
                        sell_params = custom_params['sell_params']
                    if 'minimal_roi' in custom_params:
                        custom_stop = custom_params['minimal_roi']
                    if 'custom_stop' in custom_params:
                        custom_stop = custom_params['custom_stop']
                    if 'dynamic_roi' in custom_params:
                        dynamic_roi = custom_params['dynamic_roi']
                    break

        if params == 'buy':
            return buy_params
        if params == 'sell':
            return sell_params
        if params == 'minimal_roi':
            return minimal_roi
        if params == 'custom_stop':
            return custom_stop
        if params == 'dynamic_roi':
            return dynamic_roi

        return False

    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)

            if rate:
                return rate

        ask_strategy = self.config.get('ask_strategy', {})
        if ask_strategy.get('use_order_book', False):
            ob = self.dp.orderbook(pair, 1)
            rate = ob[f"{ask_strategy['price_side']}s"][0][0]
        else:
            ticker = self.dp.ticker(pair)
            rate = ticker['last']

        self.custom_current_price_cache[pair] = rate
        return rate

    """
    Stripped down version from Schism, meant only to update the price data a bit
    more frequently than the default instead of getting all sorts of trade information
    """
    def populate_trades(self, pair: str) -> dict:

        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        trade_data = {}
        trade_data['active_trade'] = False

        if self.config['runmode'].value in ('live', 'dry_run'):

            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            if active_trade:

                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

        return trade_data

class Solipsis3_BTC(Solipsis3):

    timeframe = '15m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "720": 0.005,
        "1440": 0
    }

class Solipsis3_ETH(Solipsis3):

    timeframe = '15m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "720": 0.005,
        "1440": 0
    }
