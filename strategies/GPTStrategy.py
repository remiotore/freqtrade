import logging
from functools import reduce
import datetime
from datetime import timedelta
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from typing import Optional

logger = logging.getLogger(__name__)


class GPTStrategy(IStrategy):
    """
    The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    position_adjustment_enable = False

    stoploss = -0.04

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
    }

    max_entry_position_adjustment = 1

    max_dca_multiplier = 2

    minimal_roi = {"0": 0.03, "5000": -1}

    process_only_new_candles = True

    can_short = True

    plot_config = {
        "main_plot": {},
        "subplots": {
            "sentiment": {
                "sentiment_yes": {
                    "color": "green",
                    "type": "line"
                },
                "sentiment_no": {
                    "color": "red",
                    "type": "line"
                },
                "sentiment_unkown": {
                    "color": "blue",
                    "type": "line"
                }
            },
            "expert": {
                "expert_long_enter": {
                    "color": "green",
                    "type": "bar"
                },
                "expert_long_exit": {
                    "color": "red",
                    "type": "bar"
                },
                "expert_short_enter": {
                    "color": "gray",
                    "type": "bar"
                },
                "expert_short_exit": {
                    "color": "purple",
                    "type": "bar"
                },
                "expert_neutral": {
                    "color": "blue",
                    "type": "bar"
                }
            }
        }
    }

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            }
        ]

    use_exit_signal = True
    startup_candle_count: int = 160






    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek)
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour)
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&-empty"] = "0"
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [
            df["expert_long_enter"] == 1
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), [
                    "enter_long", "enter_tag"]
            ] = (1, f"{df['expert_opinion'].iloc[-1]}, entering long")

        enter_short_conditions = [
            df["expert_short_enter"] == 1
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), [
                    "enter_short", "enter_tag"]
            ] = (1, f"{df['expert_opinion'].iloc[-1]}, entering short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        return df

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):

        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(
            self.timeframe, (trade.open_date_utc -
                             timedelta(minutes=int(self.timeframe[:-1])))
        )
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()

        entry_tag = trade.enter_tag

        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if trade_duration > 1000:
            return "trade expired"

        if last_candle["expert_short_exit"] == 1 and entry_tag == "short":
            return f"{last_candle['expert_opinion']}, exiting short"

        if last_candle["expert_long_exit"] == 1 and entry_tag == "long":
            return f"{last_candle['expert_opinion']}, exiting long"

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        logger.info(f"{last_candle['expert_opinion']}\n\n Entering {side} on {pair} at {rate}")
        return True