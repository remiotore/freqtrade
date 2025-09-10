# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class YourStrat(IStrategy):
    # replace this by your strategy
    pass

class TrailingBuyStrat(YourStrat):
    # Orignal idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #
    # if process_only_new_candles = True, then you need to use 1m timeframe (and normal strategy timeframe as informative)
    # if process_only_new_candles = False, it will use ticker data and you won't need to change anything

    process_only_new_candles = False

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 300

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.1  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.002  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def buy(self, dataframe, pair: str, current_price: float, buy_tag: str):
        dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
        ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
        if 'buy_tag' in dataframe.columns:
            dataframe.iloc[-1, dataframe.columns.get_loc('buy_tag')] = f"{buy_tag} ({ratio} %)"
        self.trailing_buy_info(pair, current_price)
        logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if current_trailing_profit_ratio > 0 and last_candle['pre_buy'] == 1:
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.trailing_buy_info(pair, current_rate)
            self.trailing_buy(pair, reinit=True)
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        self.trailing_buy(pair, reinit=True)
        return val

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        # stop trailing when buy signal ! prevent from buying much higher price when slot is free
        self.trailing_buy_info(pair, rate)
        self.trailing_buy(pair, reinit=True)
        logger.info(f'STOP trailing buy for {pair} because I buy it')
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)
        dataframe = dataframe.rename(columns={"buy": "pre_buy"})

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):  # trailing live dry ticker, 1m
            last_candle = dataframe.iloc[-1].squeeze()
            if not self.process_only_new_candles:
                current_price = self.get_current_price(metadata["pair"])
            else:
                current_price = last_candle['close']
            dataframe['buy'] = 0
            trailing_buy = self.trailing_buy(metadata['pair'])
            trailing_buy_offset = self.trailing_buy_offset(dataframe, metadata['pair'], current_price)

            if not trailing_buy['trailing_buy_order_started'] and last_candle['pre_buy'] == 1:
                open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                if not open_trades:
                    # start trailing buy
                    self.custom_info_trail_buy[metadata["pair"]]['trailing_buy'] = {
                        'trailing_buy_order_started': True,
                        'trailing_buy_order_uplimit': last_candle['close'],
                        'start_trailing_price': last_candle['close'],
                        'buy_tag': last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal',
                        'start_trailing_time': datetime.now(timezone.utc),
                        'offset': 0,
                    }
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'start trailing buy for {metadata["pair"]} at {last_candle["close"]}')

            elif trailing_buy['trailing_buy_order_started']:
                if trailing_buy_offset == 'forcebuy':
                    # buy in custom conditions
                    self.buy(dataframe, metadata['pair'], current_price, trailing_buy['buy_tag'])
                elif trailing_buy_offset is None:
                    # stop trailing buy custom conditions
                    self.trailing_buy(metadata['pair'], reinit=True)
                    logger.info(f'STOP trailing buy for {metadata["pair"]} because "trailing buy offset" returned None')
                elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                    # update uplimit
                    old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                    self.custom_info_trail_buy[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'])
                    self.custom_info_trail_buy[metadata["pair"]]['trailing_buy']['offset'] = trailing_buy_offset
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'update trailing buy for {metadata["pair"]} at {old_uplimit} -> {self.custom_info_trail_buy[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                    # buy ! current price > uplimit && lower thant starting price
                    self.buy(dataframe, metadata['pair'], current_price, trailing_buy['buy_tag'])
                elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                    # stop trailing buy because price is too high
                    self.trailing_buy(metadata['pair'], reinit=True)
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'STOP trailing buy for {metadata["pair"]} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                else:
                    # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'price too high for {metadata["pair"]} !')
        else:  # No buy trailing
            dataframe.loc[
                (dataframe['pre_buy'] == 1)
                , 'buy'] = 1
        return dataframe

    def get_current_price(self, pair: str) -> float:
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']
        return current_price

import time
class TrailingBuyStrat2(YourStrat):
    # Orignal idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #
    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 3600
    delay_between_loop_forcebuy = 3

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.1  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'forcebuy': False,
        'start_trailing_time': None,
        'offset': 0,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def buy(self, dataframe, pair: str, current_price: float, buy_tag: str):
        dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
        ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
        if 'buy_tag' in dataframe.columns:
            new_buy_tag = f"{buy_tag} ({ratio} %)"
            dataframe.iloc[-1, dataframe.columns.get_loc('buy_tag')] = new_buy_tag
            self.custom_info_trail_buy[pair]['buy_tag'] = new_buy_tag
        self.trailing_buy_info(pair, current_price)
        logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1)):
                # more than trailing_expire_seconds, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.trailing_buy_info(pair, current_rate)
            self.trailing_buy(pair, reinit=True)
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        # if pair in self.custom_info_trail_buy:
        #     trade.buy_tag = self.custom_info_trail_buy[pair]['trailing_buy']['buy_tag']
        self.trailing_buy(pair, reinit=True)
        return val

    def trailing_steps (self, dataframe, pair, current_rate):
        val = False
        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
            if (len(dataframe) < 1):
                logger.info(f'start of trailing_steps. Dataframe empty!! Stop trailing')
            else:
                last_candle = dataframe.iloc[-1].squeeze()
                current_price = current_rate
                trailing_buy = self.trailing_buy(pair)
                trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                if not trailing_buy['trailing_buy_order_started'] :
                    # start trailing buy
                    self.custom_info_trail_buy[pair]['trailing_buy'] = {
                        'trailing_buy_order_started': True,
                        'trailing_buy_order_uplimit': last_candle['close'],
                        'start_trailing_price': last_candle['close'],
                        'buy_tag': last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal',
                        'start_trailing_time': datetime.now(timezone.utc),
                        'offset': 0,
                    }
                    self.trailing_buy_info(pair, current_price)
                    logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                elif trailing_buy['trailing_buy_order_started']:
                    if trailing_buy_offset == 'forcebuy':
                        # buy in custom conditions
                        self.buy(dataframe, pair, current_price, trailing_buy['buy_tag'])
                        val = True
                    elif trailing_buy_offset is None:
                        # stop trailing buy custom conditions
                        self.trailing_buy(pair, reinit=True)
                        logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')
                    elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                        # update uplimit
                        old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                        self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                        self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                        self.trailing_buy_info(pair, current_price)
                        if (old_uplimit != self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]):
                            logger.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                    elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                        # buy ! current price > uplimit && lower thant starting price
                        self.buy(dataframe, pair, current_price, trailing_buy['buy_tag'])
                        val = True
                    elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                        # stop trailing buy because price is too high
                        self.trailing_buy(pair, reinit=True)
                        self.trailing_buy_info(pair, current_price)
                        logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                    else:
                        # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                        self.trailing_buy_info(pair, current_price)
                        logger.info(f'price too high for {pair} !')
                
                    if val:
                        # stop trailing when buy signal ! prevent from buying much higher price when slot is free
                        self.trailing_buy_info(pair, current_price)
                        self.trailing_buy(pair, reinit=True)
                        logger.info(f'STOP trailing buy for {pair} because I buy it')
        return val

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        if rate:
            val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
            if val:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    if self.trailing_buy_order_enabled and (self.config['runmode'].value in ('live', 'dry_run')):

                        trailing_buy = self.trailing_buy(pair)
                        trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, rate)

                        last_candle = dataframe.iloc[-1].squeeze()

                        open_trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True), ]).all()
                        if not trailing_buy['trailing_buy_order_started']:
                            if not open_trades: # No open trade
                                if (last_candle['buy'] == True): # Normal buy trigger
                                    logger.info(f'Normal buy for {pair}. No current open trade. Activate trailing step')
                                    trailing_buy['forcebuy'] = False
                                else: #forcebuy
                                    trailing_buy['forcebuy'] = True
                                    logger.info(f'Forcebuy for {pair}. No current open trade. Activate trailing loops')
                            else: # there is open trade, manual DCA
                                trailing_buy['forcebuy'] = True
                                logger.info(f'Forcebuy for {pair}. There is open trade(s) for it. Manual DCA. Activate trailing loops')
                        else:
                            if open_trades: #there is open trade, manual DCA
                                trailing_buy['forcebuy'] = True
                                logger.info(f'Forcebuy for {pair}. Manual DCA. Continue trailing loops')
                            else:
                                if (last_candle['buy'] == True): #Normal buy trigger
                                    logger.info(f'Normal buy for {pair}. Continue trailing step')
                                    trailing_buy['forcebuy'] = False
                                else: #forcebuy
                                    trailing_buy['forcebuy'] = True
                                    logger.info(f'Forcebuy for {pair}. Continue trailing loops')

                        go_loop = True
                        while (go_loop):
                            val = self.trailing_steps(dataframe, pair, rate)

                            forcebuy = False
                            try:
                                forcebuy = trailing_buy['forcebuy']
                            except KeyError:
                                forcebuy = False

                            if(val or (forcebuy == False)): #Get out of the loop
                                go_loop = False
                            else:
                                rate = self.get_current_price(pair)
                                current_time = datetime.now(timezone.utc)
                                current_duration = 0
                                try:
                                    current_duration = (current_time - trailing_buy['start_trailing_time']).total_seconds()
                                except TypeError:
                                    current_duration = 0
                                finally:
                                    if (current_duration > self.trailing_expire_seconds): #Has been trailing for too long. Decide buy or not
                                        go_loop = False
                                        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, rate)
                                        if (current_trailing_profit_ratio > 0): #buy the coin
                                            val = True
                                            logger.info(f'Trailing has been too long. We might save {current_trailing_profit_ratio*100:.2f}% on this trade. Buy if there is free slot!')
                                        else:
                                            logger.info(f'No saving and trailing has been too long. Not buying')

                                        self.trailing_buy(pair, reinit=True)
                                if (go_loop):
                                    time.sleep(self.delay_between_loop_forcebuy)
                                    rate = self.get_current_price(pair)

                else:
                    logger.info(f'no prior candle data')
            else:
                logger.info(f'The trade for {pair} failed to pass base confirm_trade_entry')
        else:
            val = False
            logger.info(f'No rate for {pair}!!!')
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        last_candle = dataframe.iloc[-1].squeeze()
        if ((last_candle['buy'] == 0) and (self.trailing_buy_order_enabled) and (self.config['runmode'].value in ('live', 'dry_run'))):
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (trailing_buy['trailing_buy_order_started']):
                logger.info(f'Buy signal for {metadata["pair"]} not active, but there is active trailing from past candle. Trigger it manually')
                dataframe['buy'] = 1
        
        return dataframe

    def get_current_price(self, pair: str) -> float:
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']
        return current_price

