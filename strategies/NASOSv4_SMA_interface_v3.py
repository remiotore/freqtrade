

import logging
from datetime import datetime, timezone
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
    stoploss_from_open,
)
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame



logger = logging.getLogger(__name__)



buy_params = {
    "base_nb_candles_buy": 8,
    "ewo_high": 2.403,
    "ewo_high_2": -5.585,
    "ewo_low": -14.378,
    "lookback_candles": 3,
    "low_offset": 0.984,
    "low_offset_2": 0.942,
    "profit_threshold": 1.008,
    "rsi_buy": 72,
}

sell_params = {
    "base_nb_candles_sell": 16,
    "high_offset": 1.084,
    "high_offset_2": 1.401,
    "pHSL": -0.15,
    "pPF_1": 0.016,
    "pPF_2": 0.024,
    "pSL_1": 0.014,
    "pSL_2": 0.022,
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.SMA(df, timeperiod=ema_length)
    ema2 = ta.SMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["low"] * 100
    return emadif


class NASOSv4_SMA_TB(NASOSv4_SMA_interface_v3):











    process_only_new_candles = False

    custom_info_trail_buy = dict()

    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    perfect_enter_tags = ["ewo_low"]

    def is_perfect_enter_tag(self, enter_tag: str):
        for perfect_enter_tag in self.perfect_enter_tags:
            if enter_tag in perfect_enter_tag:
                return True
        return False

    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.1  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.002  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        "trailing_buy_order_started": False,
        "trailing_buy_order_uplimit": 0,
        "start_trailing_price": 0,
        "enter_tag": None,
        "start_trailing_time": None,
        "offset": 0,
    }

    def trailing_buy(self, pair, reinit=False):

        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if reinit or not "trailing_buy" in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]["trailing_buy"] = self.init_trailing_dict
        return self.custom_info_trail_buy[pair]["trailing_buy"]

    def trailing_buy_info(self, pair: str, current_price: float):

        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = current_time - trailing_buy["start_trailing_time"]
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : start: {trailing_buy['start_trailing_price']:.4f}, duration: {duration}, current: {current_price:.4f}, uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, profit: {self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}%, offset: {trailing_buy['offset']}"
            )

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy["trailing_buy_order_started"]:
            return (
                trailing_buy["start_trailing_price"] - current_price
            ) / trailing_buy["start_trailing_price"]
        else:
            return 0

    def buy(self, dataframe, pair: str, current_price: float, enter_tag: str):
        dataframe.iloc[-1, dataframe.columns.get_loc("enter_long")] = 1
        ratio = "%.2f" % (self.current_trailing_profit_ratio(pair, current_price) * 100)
        if "enter_tag" in dataframe.columns:
            dataframe.iloc[-1, dataframe.columns.get_loc("enter_tag")] = (
                f"{enter_tag} ({ratio} %)"
            )
        self.trailing_buy_info(pair, current_price)
        logger.info(
            f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full"
        )

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):




        current_trailing_profit_ratio = self.current_trailing_profit_ratio(
            pair, current_price
        )
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy["trailing_buy_order_started"]:
            return default_offset


        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy["start_trailing_time"]
        if self.is_perfect_enter_tag(trailing_buy["enter_tag"]):
            return "forcebuy"
        elif trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if current_trailing_profit_ratio > 0 and last_candle["pre_buy"] == 1:

                return "forcebuy"
            else:

                return None
        elif (
            self.trailing_buy_uptrend_enabled
            and trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend
            and (current_trailing_profit_ratio < -1 * self.min_uptrend_trailing_profit)
        ):

            return "forcebuy"

        if current_trailing_profit_ratio < 0:

            return default_offset
        trailing_buy_offset = {0.06: 0.02, 0.03: 0.01, 0: default_offset}

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset



    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        tag = super().custom_exit(
            pair, trade, current_time, current_rate, current_profit, **kwargs
        )
        if tag:
            self.trailing_buy_info(pair, current_rate)
            self.trailing_buy(pair, reinit=True)
            logger.info(f"STOP trailing buy for {pair} because of {tag}")
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata["pair"])
        return dataframe

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        **kwargs,
    ) -> bool:
        val = super().confirm_trade_exit(
            pair, trade, order_type, amount, rate, time_in_force, exit_reason, **kwargs
        )
        self.trailing_buy(pair, reinit=True)
        return val

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        **kwargs,
    ) -> bool:
        val = super().confirm_trade_entry(
            pair, order_type, amount, rate, time_in_force, **kwargs
        )

        self.trailing_buy_info(pair, rate)
        self.trailing_buy(pair, reinit=True)
        logger.info(f"STOP trailing buy for {pair} because I buy it")
        return val

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)

        if not self.trailing_buy_order_enabled or not self.config["runmode"].value in (
            "live",
            "dry_run",
        ):  # no buy trailing
            return dataframe

        dataframe = dataframe.rename(columns={"enter_long": "pre_buy"})
        last_candle = dataframe.iloc[-1].squeeze()
        dataframe["enter_long"] = 0
        trailing_buy = self.trailing_buy(metadata["pair"])

        if (
            not trailing_buy["trailing_buy_order_started"]
            and last_candle["pre_buy"] == 1
        ):
            current_price = self.get_current_price(metadata["pair"], last_candle)
            open_trades = Trade.get_trades(
                [Trade.pair == metadata["pair"], Trade.is_open.is_(True)]
            ).all()
            if not open_trades:

                self.custom_info_trail_buy[metadata["pair"]]["trailing_buy"] = {
                    "trailing_buy_order_started": True,
                    "trailing_buy_order_uplimit": last_candle["close"],
                    "start_trailing_price": last_candle["close"],
                    "enter_tag": (
                        last_candle["enter_tag"]
                        if "enter_tag" in last_candle
                        else "buy signal"
                    ),
                    "start_trailing_time": datetime.now(timezone.utc),
                    "offset": 0,
                }
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(
                    f"start trailing buy for {metadata['pair']} at {last_candle['close']}"
                )
        elif trailing_buy["trailing_buy_order_started"]:
            current_price = self.get_current_price(metadata["pair"], last_candle)
            trailing_buy_offset = self.trailing_buy_offset(
                dataframe, metadata["pair"], current_price
            )

            if trailing_buy_offset == "forcebuy":

                self.buy(
                    dataframe,
                    metadata["pair"],
                    current_price,
                    trailing_buy["enter_tag"],
                )
            elif trailing_buy_offset is None:

                self.trailing_buy(metadata["pair"], reinit=True)
                logger.info(
                    f"""STOP trailing buy for {metadata['pair']} because "trailing buy offset" returned None"""
                )
            elif current_price < trailing_buy["trailing_buy_order_uplimit"]:

                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                self.custom_info_trail_buy[metadata["pair"]]["trailing_buy"][
                    "trailing_buy_order_uplimit"
                ] = min(
                    current_price * (1 + trailing_buy_offset),
                    self.custom_info_trail_buy[metadata["pair"]]["trailing_buy"][
                        "trailing_buy_order_uplimit"
                    ],
                )
                self.custom_info_trail_buy[metadata["pair"]]["trailing_buy"][
                    "offset"
                ] = trailing_buy_offset
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(
                    f"update trailing buy for {metadata['pair']} at {old_uplimit} -> {self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['trailing_buy_order_uplimit']}"
                )
            elif current_price < trailing_buy["start_trailing_price"] * (
                1 + self.trailing_buy_max_buy
            ):

                self.buy(
                    dataframe,
                    metadata["pair"],
                    current_price,
                    trailing_buy["enter_tag"],
                )
            elif current_price > trailing_buy["start_trailing_price"] * (
                1 + self.trailing_buy_max_stop
            ):

                self.trailing_buy(metadata["pair"], reinit=True)
                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(
                    f"STOP trailing buy for {metadata['pair']} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}"
                )
            else:

                self.trailing_buy_info(metadata["pair"], current_price)
                logger.info(f"price too high for {metadata['pair']} !")
        return dataframe

    def get_current_price(self, pair: str, last_candle) -> float:
        if self.process_only_new_candles:
            current_price = last_candle["close"]
        else:
            ticker = self.dp.ticker(pair)
            current_price = ticker["last"]
        return current_price


class NASOSv4_SMA_interface_v3(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {



        "0": 10
    }

    stoploss = -0.15

    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params["base_nb_candles_buy"], space="buy", optimize=True
    )
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params["base_nb_candles_sell"], space="sell", optimize=True
    )
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params["low_offset"], space="buy", optimize=False
    )
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params["low_offset_2"], space="buy", optimize=False
    )
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params["high_offset"], space="sell", optimize=True
    )
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params["high_offset_2"], space="sell", optimize=True
    )

    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 24, default=buy_params["lookback_candles"], space="buy", optimize=True
    )

    profit_threshold = DecimalParameter(
        1.0, 1.03, default=buy_params["profit_threshold"], space="buy", optimize=True
    )

    ewo_low = DecimalParameter(
        -20.0, -8.0, default=buy_params["ewo_low"], space="buy", optimize=False
    )
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params["ewo_high"], space="buy", optimize=False
    )

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params["ewo_high_2"], space="buy", optimize=False
    )

    rsi_buy = IntParameter(
        50, 100, default=buy_params["rsi_buy"], space="buy", optimize=False
    )


    pHSL = DecimalParameter(
        -0.2, -0.04, default=-0.15, decimals=3, space="sell", optimize=False, load=True
    )

    pPF_1 = DecimalParameter(
        0.008, 0.02, default=0.016, decimals=3, space="sell", optimize=False, load=True
    )
    pSL_1 = DecimalParameter(
        0.008, 0.02, default=0.014, decimals=3, space="sell", optimize=False, load=True
    )

    pPF_2 = DecimalParameter(
        0.04, 0.1, default=0.024, decimals=3, space="sell", optimize=False, load=True
    )
    pSL_2 = DecimalParameter(
        0.02, 0.07, default=0.022, decimals=3, space="sell", optimize=False, load=True
    )

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    order_time_in_force = {"entry": "gtc", "exit": "ioc"}

    timeframe = "5m"
    inf_1h = "1h"

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = True

    plot_config = {
        "main_plot": {
            "ma_buy": {"color": "orange"},
            "ma_sell": {"color": "orange"},
        },
    }

    slippage_protection = {"retries": 3, "max_slippage": -0.02}


    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value




        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + (current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1)
        else:
            sl_profit = HSL



        return stoploss_from_open(sl_profit, current_profit)

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if last_candle is not None:
            if exit_reason in ["exit_signal"]:
                if (
                    last_candle["hma_50"] * 1.149 > last_candle["ema_100"]
                    and last_candle["close"] < last_candle["ema_100"] * 0.951
                ):  # *1.2
                    return False

        try:
            state = self.slippage_protection["__pair_retries"]
        except KeyError:
            state = self.slippage_protection["__pair_retries"] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = rate / candle["close"] - 1
        if slippage < self.slippage_protection["max_slippage"]:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection["retries"]:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.inf_1h
        )










        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in self.base_nb_candles_buy.range:
            dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        dataframe["hma_50"] = qtpylib.hull_moving_average(dataframe["close"], window=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)

        dataframe["sma_9"] = ta.SMA(dataframe, timeperiod=9)

        dataframe["EWO"] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True
        )

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(

            dataframe["close_1h"].rolling(self.lookback_candles.value).max()
            < dataframe["close"] * self.profit_threshold.value
        )

        dataframe.loc[
            (dataframe["rsi_fast"] < 35)
            & (
                dataframe["close"]
                < dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                * self.low_offset.value
            )
            & (dataframe["EWO"] > self.ewo_high.value)
            & (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["volume"] > 0)
            & (
                dataframe["close"]
                < dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                * self.high_offset.value
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ewo1")
        dataframe.loc[
            (dataframe["rsi_fast"] < 35)
            & (
                dataframe["close"]
                < dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                * self.low_offset_2.value
            )
            & (dataframe["EWO"] > self.ewo_high_2.value)
            & (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["volume"] > 0)
            & (
                dataframe["close"]
                < dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                * self.high_offset.value
            )
            & (dataframe["rsi"] < 25),
            ["enter_long", "enter_tag"],
        ] = (1, "ewo2")
        dataframe.loc[
            (dataframe["rsi_fast"] < 35)
            & (
                dataframe["close"]
                < dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                * self.low_offset.value
            )
            & (dataframe["EWO"] < self.ewo_low.value)
            & (dataframe["volume"] > 0)
            & (
                dataframe["close"]
                < dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                * self.high_offset.value
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ewolow")

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "enter_long"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            (dataframe["close"] > dataframe["sma_9"])
            & (
                dataframe["close"]
                > dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                * self.high_offset_2.value
            )
            & (dataframe["rsi"] > 50)
            & (dataframe["volume"] > 0)
            & (dataframe["rsi_fast"] > dataframe["rsi_slow"])
            | (dataframe["close"] < dataframe["hma_50"])
            & (
                dataframe["close"]
                > dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                * self.high_offset.value
            )
            & (dataframe["volume"] > 0)
            & (dataframe["rsi_fast"] > dataframe["rsi_slow"])
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "exit_long"] = 1

        return dataframe
