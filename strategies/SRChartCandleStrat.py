import datetime
import logging
from datetime import datetime
from datetime import timedelta, timezone
from functools import reduce
from typing import Optional
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from talib import CDLDOJI
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


class Elliot:
    base_nb_candles_sell = 22
    base_nb_candles_buy = 12
    ewo_low = 10.289
    ewo_high = 3.001
    fast_ewo = 50
    slow_ewo = 200
    low_offset = 0.987
    rsi_buy = 58
    high_offset = 1.014
    buy_ema_cofi = 0.97
    buy_fastk = 20
    buy_fastd = 20
    buy_adx = 30
    buy_ewo_high = 3.55

    def use_hyperopts(self, base_nb_candles_sell, base_nb_candles_buy, low_offset, ewo_high, ewo_low, rsi_buy,
                      high_offset, buy_ema_cofi, buy_fastk, buy_fastd, buy_adx, buy_ewo_high):
        self.base_nb_candles_sell = base_nb_candles_sell
        self.base_nb_candles_buy = base_nb_candles_buy
        self.low_offset = low_offset
        self.ewo_high = ewo_high
        self.ewo_low = ewo_low
        self.rsi_buy = rsi_buy
        self.high_offset = high_offset
        self.buy_ema_cofi = buy_ema_cofi
        self.buy_fastk = buy_fastk
        self.buy_fastd = buy_fastd
        self.buy_adx = buy_adx
        self.buy_ewo_high = buy_ewo_high

    def populate_indicators(self, dataframe: DataFrame):
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['EWO'] = self.__EWO(dataframe, self.fast_ewo, self.slow_ewo)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        return dataframe

    def populate_entry_trend_v1(self, dataframe: DataFrame, conditions: list):
        buy1ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'enter_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)
        return (dataframe, conditions)

    def populate_entry_trend_v2(self, dataframe: DataFrame, conditions: list):
        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'enter_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)
        return (dataframe, conditions)

    def populate_entry_trend_cofi(self, dataframe: DataFrame, conditions: list):
        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi_'
        conditions.append(is_cofi)
        return (dataframe, conditions)

    def __EWO(self, dataframe: DataFrame, ema_length=5, ema2_length=3):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        return (ema1 - ema2) / df['close'] * 100


class SRChartCandleStrat(IStrategy):
    INTERFACE_VERSION = 3
    max_safety_orders = 3
    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    pairs_close_to_high = []
    support_dict = {}
    resistance_dict = {}
    out_open_trades_limit = 10
    stoploss = -0.9

    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    minimal_roi = {
        "0": 0.5,
        "120": 0.3,
        "240": 0.1,
        "360": 0.07,
        "480": 0.05,
        "720": 0.03,
        "960": 0.01,
        "1440": 0.005,
        "2880": 0.003,
        "4320": 0.001,
        "5760": 0.000
    }
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    timeframed_drops = {
        '1m': -0.01,
        '5m': -0.05,
        '15m': -0.05,
        '30m': -0.075,
        '1h': -0.1
    }
    timeframes_in_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    elliot = Elliot()
    base_nb_candles_sell = IntParameter(8, 20, default=elliot.base_nb_candles_sell, space='sell', optimize=False)
    base_nb_candles_buy = IntParameter(8, 20, default=elliot.base_nb_candles_buy, space='buy', optimize=False)
    low_offset = DecimalParameter(0.975, 0.995, default=elliot.low_offset, space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 5, default=elliot.ewo_high, space='buy', optimize=True)
    ewo_low = DecimalParameter(-20.0, -7.0, default=-elliot.ewo_low, space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=elliot.rsi_buy, space='buy', optimize=False)
    high_offset = DecimalParameter(1.000, 1.010, default=elliot.high_offset, space='sell', optimize=True)
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=elliot.buy_ema_cofi, optimize=True)
    buy_fastk = IntParameter(20, 30, default=elliot.buy_fastk, optimize=True)
    buy_fastd = IntParameter(20, 30, default=elliot.buy_fastd, optimize=True)
    buy_adx = IntParameter(20, 30, default=elliot.buy_adx, optimize=True)
    buy_ewo_high = DecimalParameter(2, 12, default=elliot.buy_ewo_high, optimize=True)
    elliot.use_hyperopts(base_nb_candles_sell, base_nb_candles_buy, low_offset, ewo_high, ewo_low, rsi_buy, high_offset,
                         buy_ema_cofi, buy_fastk, buy_fastd, buy_adx, buy_ewo_high)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    def version(self) -> str:
        return "SRChartCandleStrat v1.0"

    def dynamic_stop_loss_take_profit(self, dataframe: DataFrame) -> DataFrame:
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe['stop_loss'] = dataframe['low'].shift(1) - atr.shift(1) * 0.8
        dataframe['take_profit'] = dataframe['high'].shift(1) + atr.shift(1) * 2.5
        return dataframe

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if current_rate is None:
            return None

        pct_threshold = self.timeframed_drops[self.timeframe]

        current_time = datetime.utcnow()

        try:

            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()
        except Exception as e:

            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        if self.is_pair_locked(pair=trade.pair):
            return None

        last_candle = df.iloc[-1]

        if trade.pair not in self.lowest_prices:
            self.lowest_prices[trade.pair] = trade.open_rate

        if trade.pair not in self.price_drop_percentage:
            self.price_drop_percentage[trade.pair] = {"last_drop_time": current_time, "last_drop_rate": current_rate}

        if current_rate < self.lowest_prices[trade.pair]:
            self.lowest_prices[trade.pair] = current_rate

        price_drop = (self.lowest_prices[trade.pair] - trade.open_rate) / trade.open_rate

        if ((price_drop <= pct_threshold)
                and (self.price_drop_percentage[trade.pair].get("last_drop_time", current_time) != current_time)):

            if "last_drop_rate" in self.price_drop_percentage[trade.pair].keys():
                last = self.price_drop_percentage[trade.pair].get("last_drop_rate")
                if last is not None:
                    if current_rate < last:
                        self.price_drop_percentage[trade.pair]["last_drop_time"] = current_time
                        self.price_drop_percentage[trade.pair]["last_drop_rate"] = current_rate

            if self.price_drop_percentage[trade.pair].get("last_drop_time") > current_time:
                time_since_last_drop = current_time - self.price_drop_percentage[trade.pair]["last_drop_time"]
                if time_since_last_drop.total_seconds() / 3600 >= 3:
                    logging.info(f"Locking {trade.pair}")
                    self.lock_pair(trade.pair, until=datetime.now(timezone.utc) + timedelta(
                        minutes=8 * 60), reason='STILLDROP_LOCK')

                    return None  # Avoid further DCA

        last_buy_order = None
        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in trade.orders)
        for order in reversed(trade.orders):
            if order.ft_order_side == 'buy' and order.status == 'closed':
                last_buy_order = order
                break

        if trade.pair not in self.price_drop_percentage:
            self.price_drop_percentage[trade.pair] = {"last_drop_time": None, "last_drop_rate": None}

        if self.max_safety_orders >= count_of_buys:

            pct_diff = self.calculate_percentage_difference(original_price=last_buy_order.price,
                                                            current_price=current_rate)

            if pct_diff < pct_threshold:
                if last_buy_order and current_rate < last_buy_order.price:

                    rsi_value = last_candle['rsi']  # Assuming RSI is part of the dataframe
                    w_rsi = last_candle['weighted_rsi']  # Assuming weighted RSI is part of the dataframe
                    if rsi_value <= w_rsi:



                        total_stake_amount = self.wallets.get_total_stake_amount()

                        calculated_dca_stake = self.calculate_dca_price(base_value=trade.stake_amount,
                                                                        decline=current_profit * 100,
                                                                        target_percent=1)

                        while calculated_dca_stake >= total_stake_amount:
                            calculated_dca_stake = calculated_dca_stake / 4



                        return calculated_dca_stake

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.prepare(dataframe=dataframe)
        self.dynamic_stop_loss_take_profit(dataframe=dataframe)
        self.prepare_adx(dataframe=dataframe)
        self.prepare_rsi(dataframe=dataframe)
        self.prepare_stochastic(dataframe=dataframe)
        self.prepare_ema_diff_buy_signal(dataframe=dataframe)
        self.prepare_sma(dataframe=dataframe)
        self.prepare_ewo(dataframe=dataframe)
        self.prepare_doji(dataframe=dataframe)
        self.prepare_fibs(dataframe=dataframe)
        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)
        dataframe = self.elliot.populate_indicators(dataframe=dataframe)
        return dataframe
        pass

    def populate_entry_trend_sr(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if metadata['pair'] in self.support_dict and metadata['pair'] in self.resistance_dict:
            supports = self.support_dict[metadata['pair']]
            resistances = self.resistance_dict[metadata['pair']]

            if supports and resistances:

                dataframe['nearest_support'] = dataframe['close'].apply(
                    lambda x: min([support for support in supports if support <= x], default=x,
                                  key=lambda support: abs(x - support))
                )
                dataframe['nearest_resistance'] = dataframe['close'].apply(
                    lambda x: min([resistance for resistance in resistances if resistance >= x], default=x,
                                  key=lambda resistance: abs(x - resistance))
                )

                dataframe['distance_to_support_pct'] = (dataframe['nearest_support'] - dataframe['close']) / dataframe[
                    'close'] * 100
                dataframe['distance_to_resistance_pct'] = (dataframe['nearest_resistance'] - dataframe['close']) / \
                                                          dataframe['close'] * 100

                buy_threshold = 0.1  # 0.1 %
                dataframe.loc[
                    (dataframe['distance_to_support_pct'] >= 0) &
                    (dataframe['distance_to_support_pct'] <= buy_threshold) &
                    (dataframe['distance_to_resistance_pct'] >= buy_threshold),
                    'buy_signal'
                ] = 1

                dataframe.loc[
                    (dataframe['distance_to_support_pct'] >= 0) &
                    (dataframe['distance_to_support_pct'] <= buy_threshold) &
                    (dataframe['distance_to_resistance_pct'] >= buy_threshold),
                    'enter_tag'
                ] += 'sr_buy_mid'

                dataframe.drop(
                    ['nearest_support', 'nearest_resistance', 'distance_to_support_pct', 'distance_to_resistance_pct'],
                    axis=1, inplace=True)

        dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_ema'] = 1
        dataframe.loc[
            (dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'enter_tag'] += 'ema_dbs_'

        dataframe.loc[(dataframe['buy_signal'] == 1) & (dataframe['buy_ema'] == 1) & (
                dataframe['rsi'] <= dataframe['weighted_rsi']), 'enter_long'] = 1

        if 'buy_support' in dataframe.columns:
            dataframe.drop(['buy_support'], axis=1, inplace=True)
        if 'buy_ema' in dataframe.columns:
            dataframe.drop(['buy_ema'], axis=1, inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if dataframe is not None:

            last_candle = dataframe.iloc[-1]
            if last_candle['doji_candle'] == 1:
                logging.info(f"Doji detected on {metadata['pair']}")
                if not self.is_pair_locked(pair=metadata['pair']):
                    self.lock_pair(pair=metadata['pair'], until=datetime.now(timezone.utc) + timedelta(
                        minutes=self.timeframe_to_minutes(self.timeframe) * 14), reason='DOJI_LOCK')
                return dataframe

            (dataframe, conditions) = self.elliot.populate_entry_trend_v1(dataframe, conditions)
            (dataframe, conditions) = self.elliot.populate_entry_trend_v2(dataframe, conditions)
            (dataframe, conditions) = self.elliot.populate_entry_trend_cofi(dataframe, conditions)

            dataframe = self.populate_entry_trend_sr(dataframe=dataframe, metadata=metadata)

            if metadata['pair'] in self.support_dict:
                s = self.support_dict[metadata['pair']]
                if s:

                    dataframe['nearest_support'] = dataframe['close'].apply(
                        lambda x: min([support for support in s if support <= x], default=x,
                                      key=lambda support: abs(x - support))
                    )

                    if 'nearest_support' in dataframe.columns:

                        dataframe['distance_to_support_pct'] = (dataframe['nearest_support'] - dataframe['close']) / \
                                                               dataframe['close'] * 100

                        buy_threshold = 0.1  # 0.1 %
                        dataframe.loc[
                            (dataframe['distance_to_support_pct'] >= 0) &
                            (dataframe['distance_to_support_pct'] <= buy_threshold),
                            'buy_support'
                        ] = 1

                        dataframe.loc[
                            (dataframe['distance_to_support_pct'] >= 0) &
                            (dataframe['distance_to_support_pct'] <= buy_threshold),
                            'enter_tag'
                        ] += 'sr_buy'

                        dataframe.drop(['nearest_support', 'distance_to_support_pct'],
                                       axis=1, inplace=True)

            dataframe.loc[(dataframe['volume'] > 0) & (dataframe['ema_diff_buy_signal'].astype(int) > 0), 'buy_ema'] = 1
            dataframe.loc[
                (dataframe['volume'] > 0) & (
                        dataframe['ema_diff_buy_signal'].astype(int) > 0), 'enter_tag'] += 'ema_dbs_'

            dataframe.loc[(dataframe['buy_support'] == 1) & (dataframe['buy_ema'] == 1) & (
                    dataframe['rsi'] <= dataframe['weighted_rsi']), 'enter_long'] = 1

            if 'buy_support' in dataframe.columns:
                dataframe.drop(['buy_support'], axis=1, inplace=True)
            if 'buy_ema' in dataframe.columns:
                dataframe.drop(['buy_ema'], axis=1, inplace=True)

            dont_buy_conditions = [
                (dataframe['enter_long'].shift(1) == 1 & (dataframe['sma_2'].shift(1) < dataframe['sma_2']))
            ]

            if conditions:
                final_condition = reduce(lambda x, y: x | y, conditions)
                dataframe.loc[final_condition, 'enter_long'] = 1
            if dont_buy_conditions:
                for condition in dont_buy_conditions:
                    dataframe.loc[condition, 'enter_long'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['fib_618']) &
                    (dataframe['sma_50'].shift(1) > dataframe['sma_50'])
            ),
            'exit_tag'] = 'fib_618_sma_50'

        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['fib_618']) &
                    (dataframe['sma_50'].shift(1) > dataframe['sma_50'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['stop_loss'].shift(1)) |
                    (dataframe['close'] > dataframe['take_profit'].shift(1))
            ),
            'exit_tag'] = 'psl'

        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['stop_loss'].shift(1)) |
                    (dataframe['close'] > dataframe['take_profit'].shift(1))
            ),
            'exit_long'] = 1

        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:







        result = Trade.get_open_trade_count() < self.out_open_trades_limit
        return result
        pass

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.15 and (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        exit_reason = f"{exit_reason}_{trade.enter_tag}"

        if 'unclog' in exit_reason or 'force' in exit_reason:

            return True

        current_profit = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if current_profit >= 0.005 and 'psl' in exit_reason:
            logging.info(f"CTE - PSL EXIT: {pair}, {current_profit}, {rate}, {exit_reason}, {amount}")
            return True

        last_candle = dataframe.iloc[-1]

        if last_candle['high'] > last_candle['open']:

            return False

        if current_profit <= 0.005:
            return False

        ema_8_current = dataframe['ema_8'].iat[-1]
        ema_14_current = dataframe['ema_14'].iat[-1]

        ema_8_previous = dataframe['ema_8'].iat[-2]
        ema_14_previous = dataframe['ema_14'].iat[-2]

        diff_current = abs(ema_8_current - ema_14_current)
        diff_previous = abs(ema_8_previous - ema_14_previous)

        diff_change_pct = (diff_previous - diff_current) / diff_previous

        if current_profit >= 0.0025:
            if ema_8_current <= ema_14_current and diff_change_pct >= 0.025:

                return True
            elif ema_8_current > ema_14_current and diff_current > diff_previous:

                return False
            else:

                return True
        else:
            return False

    pass

    def prepare(self, dataframe: DataFrame):
        if 'enter_tag' not in dataframe.columns:
            dataframe.loc[:, 'enter_tag'] = ''
        if 'exit_tag' not in dataframe.columns:
            dataframe.loc[:, 'exit_tag'] = ''
        if 'enter_long' not in dataframe.columns:
            dataframe.loc[:, 'enter_long'] = 0
        if 'exit_long' not in dataframe.columns:
            dataframe.loc[:, 'exit_long'] = 0
        pass

    def prepare_rsi(self, dataframe):
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        weights = np.linspace(1, 0, 300)  # Weights from 1 (newest) to 0 (oldest)
        weights /= weights.sum()  # Normalizing the weights so that their sum is 1
        dataframe['weighted_rsi'] = dataframe['rsi'].rolling(window=300).apply(
            lambda x: np.sum(weights * x[-300:]), raw=False
        )
        pass

    def prepare_stochastic(self, dataframe):
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        pass

    def prepare_ema_diff_buy_signal(self, dataframe):
        ema_8 = ta.EMA(dataframe, timeperiod=8)
        ema_14 = ta.EMA(dataframe, timeperiod=14)


        condition = ema_8 > ema_14
        percentage_difference = 100 * (ema_8 - ema_14).abs() / ema_14
        ema_pct_diff = percentage_difference.where(condition, -percentage_difference)
        prev_ema_pct_diff = ema_pct_diff.shift(1)
        crossover_up = (ema_8.shift(1) < ema_14.shift(1)) & (ema_8 > ema_14)
        close_to_crossover_up = (ema_8 < ema_14) & (ema_8.shift(1) < ema_14.shift(1)) & (ema_8 > ema_8.shift(1))
        ema_buy_signal = ((ema_pct_diff < 0) & (prev_ema_pct_diff < 0) & (ema_pct_diff.abs() < prev_ema_pct_diff.abs()))
        dataframe['ema_diff_buy_signal'] = (
                (ema_buy_signal | crossover_up | close_to_crossover_up) & (dataframe['rsi'] <= 55) & (
                dataframe['volume'] > 0))
        dataframe['ema_8'] = ema_8
        dataframe['ema_14'] = ema_14
        pass

    def prepare_sma(self, dataframe):
        dataframe['sma_2'] = ta.SMA(dataframe, timeperiod=2)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        pass

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

    def timeframe_to_minutes(self, timeframe):
        """Converts the timeframe to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError("Unknown timeframe: {}".format(timeframe))

    def prepare_ewo(self, dataframe):
        dataframe['EWO'] = self.__EWO(dataframe, Elliot.fast_ewo, Elliot.slow_ewo)
        pass

    def __EWO(self, dataframe: DataFrame, ema_length=5, ema2_length=3):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        return (ema1 - ema2) / df['close'] * 100

    def prepare_doji(self, dataframe):
        dataframe['doji_candle'] = (
                CDLDOJI(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close']) > 0).astype(int)
        pass

    def calculate_support_resistance_dicts(self, pair: str, df: DataFrame):
        try:
            df = self.calculate_support_resistance(df)
            self.support_dict[pair] = self.calculate_dynamic_clusters(df['support'].dropna().tolist(), 4)
            self.resistance_dict[pair] = self.calculate_dynamic_clusters(df['resistance'].dropna().tolist(), 4)
        except Exception as ex:
            logging.error(str(ex))

    def pivot_points(self, high, low, period=10):
        pivot_high = high.rolling(window=2 * period + 1, center=True).max()
        pivot_low = low.rolling(window=2 * period + 1, center=True).min()
        return high == pivot_high, low == pivot_low

    def calculate_support_resistance(self, df, period=10, loopback=290):
        high_pivot, low_pivot = self.pivot_points(df['high'], df['low'], period)
        df['resistance'] = df['high'][high_pivot]
        df['support'] = df['low'][low_pivot]
        return df

    def calculate_dynamic_clusters(self, values, max_clusters):
        """
        Dynamically calculates the averaged clusters from the given list of values.

         Args:
         values (list): List of values to cluster.
         max_clusters (int): Maximum number of clusters to create.

         Returns:
         list: List of average values for each cluster created.
        """

        def cluster_values(threshold):
            sorted_values = sorted(values)
            clusters = []
            current_cluster = [sorted_values[0]]

            for value in sorted_values[1:]:
                if value - current_cluster[-1] <= threshold:
                    current_cluster.append(value)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [value]

            clusters.append(current_cluster)
            return clusters

        threshold = 0.3  # Initial threshold value
        while True:
            clusters = cluster_values(threshold)
            if len(clusters) <= max_clusters:
                break
            threshold += 0.3

        cluster_averages = [round(sum(cluster) / len(cluster), 2) for cluster in clusters]
        return cluster_averages

    def prepare_fibs(self, dataframe):
        high_max = dataframe['high'].rolling(window=30).max()
        low_min = dataframe['low'].rolling(window=30).min()
        diff = high_max - low_min
        dataframe['fib_236'] = high_max - 0.236 * diff
        dataframe['fib_382'] = high_max - 0.382 * diff
        dataframe['fib_500'] = high_max - 0.500 * diff
        dataframe['fib_618'] = high_max - 0.618 * diff
        dataframe['fib_786'] = high_max - 0.786 * diff
        pass

    def prepare_adx(self, dataframe):
        dataframe['adx'] = ta.ADX(dataframe)
        pass
