import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta  # type: ignore
from freqtrade.strategy import IStrategy
import technical.indicators as ftt
from pandas import DataFrame
from freqtrade.strategy import IntParameter, DecimalParameter, merge_informative_pair
from datetime import datetime, timedelta
from freqtrade.persistence import Trade


class Trendhere2(IStrategy):
    """
    Adaptive Scalping Strategy for Freqtrade.

    This strategy analyzes market conditions (trending vs. ranging) in the 15m timeframe
    and adapts its entry signals accordingly. Uses Ichimoku, RSI, MFI, KAMA, and potentially other
    indicators based on trend status. Includes dynamic stoploss and dynamic leverage.
    """

    # Strategy Settings
    timeframe = "5m"
    informative_timeframe = "15m"
    can_short = True  # enables shorting

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.3,
        "50": 0.2,
        "100": 0.1,
        "200": 0
    }

    # Stoploss for the strategy.
    stoploss = -0.25  # Initial stoploss, will be dynamically adjusted
    use_custom_stoploss = False # always use custom stoploss

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02  # 1%
    trailing_stop_positive_offset = 0.25  # 2%
    trailing_only_offset_is_reached = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200  # Increased to ensure enough 15m candles

    # Dynamic Leverage Parameters
    leverage_limit = 20
    leverage_trigger = 0.01  # percentage variation that triggers a leverage change

    # Hyperparameters
    buy_rsi_high = IntParameter(30, 80, default=70, space="buy", optimize=True)
    buy_rsi_low = IntParameter(20, 50, default=30, space="buy", optimize=True)
    sell_rsi_high = IntParameter(50, 80, default=70, space="sell", optimize=True)
    sell_rsi_low = IntParameter(20, 50, default=30, space="sell", optimize=True)
    mfi_high = IntParameter(50, 80, default=70, space="buy", optimize=True)
    mfi_low = IntParameter(20, 50, default=30, space="buy", optimize=True)
    ichimoku_conversion_period = IntParameter(5, 20, default=9, space="buy", optimize=True)
    ichimoku_base_period = IntParameter(20, 50, default=26, space="buy", optimize=True)
    ichimoku_lagging_span_period = IntParameter(40, 80, default=52, space="buy", optimize=True)
    ichimoku_displacement = IntParameter(20, 50, default=26, space="buy", optimize=True)
    kama_period = IntParameter(5, 20, default=10, space="buy", optimize=True)
    atr_period = IntParameter(10, 20, default=14, space="sell", optimize=True)
    atr_mult = DecimalParameter(1.0, 3.0, default=2.0, space="stoploss", optimize=True)
    kama_close_range = DecimalParameter(0.005, 0.02, default=0.01, space="buy", optimize=True) # range of close price from kama line
    kama_range = DecimalParameter(0.0025, 0.015, default=0.005, space="buy", optimize=True) # range of kama line from previous kama line.
    stoploss_range = DecimalParameter(0.005, 0.02, default=0.01, space="stoploss", optimize=True)


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculates and populates all indicators for the strategy.

        Args:
            dataframe (DataFrame): The original dataframe
            metadata (dict): Metadata information about the pair.

        Returns:
            DataFrame: The dataframe with calculated indicators
        """
        # ----------------------------------------------------------------------------------------
        # Informative Dataframe (15m) Calculation for Trend Identification
        # ----------------------------------------------------------------------------------------
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe
        # 15min
        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        # Average Directional Index (ADX) for trend strength
        informative["adx"] = ta.ADX(informative, timeperiod=14)
        informative["adx_smooth"] = ta.SMA(informative["adx"], timeperiod=7)

        # Simple moving average
        informative["sma_50"] = ta.EMA(informative, timeperiod=50)
        informative["sma_200"] = ta.SMA(informative, timeperiod=200)
        informative["price_vs_200"] = (
            informative["close"] - informative["sma_200"]
        ) / informative["sma_200"]

        # Merge indicators from 15m timeframe with 5m dataframe
        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True
        )

        # ----------------------------------------------------------------------------------------
        # Main Dataframe (5m) Indicators Calculation
        # ----------------------------------------------------------------------------------------
        # Ichimoku Cloud
        ichimoku = ftt.ichimoku(
            dataframe,
            conversion_line_period=self.ichimoku_conversion_period.value,
            base_line_periods=self.ichimoku_base_period.value,
            laggin_span=self.ichimoku_lagging_span_period.value,
            displacement=self.ichimoku_displacement.value,
        )
        dataframe["tenkan_sen"] = ichimoku["tenkan_sen"]
        dataframe["kijun_sen"] = ichimoku["kijun_sen"]
        dataframe["senkou_a"] = ichimoku["senkou_span_a"]
        dataframe["senkou_b"] = ichimoku["senkou_span_b"]
        dataframe["leading_senkou_span_a"] = ichimoku["leading_senkou_span_a"]
        dataframe["leading_senkou_span_b"] = ichimoku["leading_senkou_span_b"]
        dataframe["ichimoku_cloud_green"] = ichimoku["cloud_green"].astype(int)
        dataframe["ichimoku_cloud_red"] = ichimoku["cloud_red"].astype(int)

        # RSI, MFI, KAMA
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["kama"] = ta.KAMA(dataframe, timeperiod=self.kama_period.value)
        dataframe["kama_prev"] = dataframe["kama"].shift(1)
        dataframe["close_prev"] = dataframe["close"].shift(1)

        # ATR for dynamic stoploss
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determines entry signals (long and short) based on market analysis.

        Args:
            dataframe (DataFrame): The dataframe with calculated indicators
            metadata (dict): Metadata information about the pair

        Returns:
            DataFrame: The dataframe with buy/sell entry signals
        """
        # Trend identification based on 15m data
        trend_threshold = 25  # Minimum ADX to be considered a trending market
        is_trending = dataframe["adx_smooth_15m"] > trend_threshold
        is_uptrend = (dataframe["sma_50_15m"] > dataframe["sma_200_15m"]) & (
            dataframe["price_vs_200_15m"] > 0.01
        )
        is_downtrend = (dataframe["sma_50_15m"] < dataframe["sma_200_15m"]) & (
            dataframe["price_vs_200_15m"] < -0.01
        )

        # -----------------------------------------------------------------------------------------------------
        # Ranging Market Entry
        # -----------------------------------------------------------------------------------------------------
        # dataframe.loc[
        #     (~is_trending)
        #     # & (dataframe["close"] > dataframe["senkou_a"])
        #     # & (dataframe["close"] > dataframe["senkou_b"])
        #     & (dataframe["rsi"] < self.buy_rsi_low.value)
        #     & (dataframe["mfi"] < self.mfi_low.value)
        #     & (abs(dataframe["close"] - dataframe["kama"]) < (dataframe["close"] * self.kama_close_range.value))
        #     & (dataframe["close"] > dataframe["kama"]),
        #     ["enter_long", "enter_tag"],
        # ] = (1, "LR")

        # dataframe.loc[
        #     (~is_trending)
        #     # & (dataframe["close"] < dataframe["senkou_a"])
        #     # & (dataframe["close"] < dataframe["senkou_b"])
        #     & (dataframe["rsi"] > self.sell_rsi_high.value)
        #     & (dataframe["mfi"] > self.mfi_high.value)
        #     & (abs(dataframe["close"] - dataframe["kama"]) < (dataframe["close"] * self.kama_close_range.value))
        #     & (dataframe["close"] < dataframe["kama"]),
        #     ["enter_short", "enter_tag"],
        # ] = (1, "SR")

        # -----------------------------------------------------------------------------------------------------
        # Trending Market Entry
        # -----------------------------------------------------------------------------------------------------
        # uptrend entry
        dataframe.loc[
            is_trending
            & is_uptrend
            & (dataframe["close"] > dataframe["kijun_sen"])
            & (dataframe["close"] > dataframe["senkou_a"])
            & (dataframe["close"] > dataframe["senkou_b"])
            & (dataframe["close"] > dataframe["kama"])
            & (abs(dataframe["close"] - dataframe["kama"]) < (dataframe["close"] * self.kama_close_range.value))
            & ((dataframe["kama"] - dataframe["kama_prev"]) > (dataframe["kama"] * self.kama_range.value)),
            ["enter_long", "enter_tag"],
        ] = (1, "LT")
        # downtrend entry
        dataframe.loc[
            is_trending
            & is_downtrend
            & (dataframe["close"] < dataframe["kijun_sen"])
            & (dataframe["close"] < dataframe["senkou_a"])
            & (dataframe["close"] < dataframe["senkou_b"])
            & (dataframe["close"] < dataframe["kama"])
            & (abs(dataframe["close"] - dataframe["kama"]) < (dataframe["close"] * self.kama_close_range.value))
            & ((dataframe["kama_prev"] - dataframe["kama"]) > (dataframe["kama"] * self.kama_range.value)),
            ["enter_short", "enter_tag"],
        ] = (1, "ST")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Determines exit signals (long and short) based on market analysis.

        Args:
            dataframe (DataFrame): The dataframe with calculated indicators
            metadata (dict): Metadata information about the pair.

        Returns:
            DataFrame: The dataframe with buy/sell exit signals
        """
        # Trend identification based on 15m data
        trend_threshold = 25  # Minimum ADX to be considered a trending market
        is_trending = dataframe["adx_smooth_15m"] > trend_threshold

        # -----------------------------------------------------------------------------------------------------
        # Ranging Market Exit
        # -----------------------------------------------------------------------------------------------------
        dataframe.loc[
            (~is_trending) & (dataframe["rsi"] > self.sell_rsi_high.value),
            "exit_long",
        ] = 1
        dataframe.loc[
            (~is_trending) & (dataframe["rsi"] < self.sell_rsi_low.value),
            "exit_short",
        ] = 1

        # -----------------------------------------------------------------------------------------------------
        # Trending Market Exit
        # -----------------------------------------------------------------------------------------------------
        dataframe.loc[
            (is_trending) & (dataframe["close"] < dataframe["kama"]),
            "exit_long",
        ] = 1

        dataframe.loc[
            (is_trending) & (dataframe["close"] > dataframe["kama"]),
            "exit_short",
        ] = 1
        
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Dynamically adjusts the stoploss based on volatility (ATR) and current trade.

        Args:
            pair (str): The trading pair.
            trade (Trade): The current trade object.
            current_time (datetime): The current time.
            current_rate (float): The current price.
            current_profit (float): The current profit/loss (can be negative).

        Returns:
            float: The new stoploss.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        if dataframe is None or len(dataframe) == 0:
            return self.stoploss  # Fallback to default stoploss

        atr = dataframe["atr"].iloc[-1]  # Get the latest ATR
        
        if trade.entry_side == "buy":
            stoploss = (dataframe["close_prev"].iloc[-1] - (dataframe["close_prev"].iloc[-1] * self.stoploss_range.value) )
        else:
            stoploss = (dataframe["close_prev"].iloc[-1] + (dataframe["close_prev"].iloc[-1] * self.stoploss_range.value) )

        return max(self.stoploss, stoploss)

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        """
        Dynamically adjusts leverage based on market condition.

         Args:
            pair (str): The trading pair.
            current_time (datetime): The current time.
            current_rate (float): The current price.
            proposed_leverage (float): The proposed leverage based on the user defined leverage
            trade_direction (str): The trade direction (long or short).

        Returns:
            float: The new leverage.
        """
        # df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        # if df is None or len(df) == 0:
        #     return proposed_leverage #Fallback to default leverage

        # if df["adx_smooth_15m"].iloc[-1] > 25 : # if is trending
        #     return min(proposed_leverage, self.leverage_limit)
        # else:
        #     return proposed_leverage / 2  # Reduce leverage for ranging markets
        return 10.0