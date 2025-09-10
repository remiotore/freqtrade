# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime

import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy
import talib.abstract as ta
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# If you want to use pandas_ta, un-comment:
import pandas_ta as pta

class AllCandlestickPatternsStrategyHyperoptv7(IStrategy):
    """
    Freqtrade strategy that integrates multiple candlestick patterns and
    a comprehensive range of technical indicators, including Overlap
    Studies, Momentum, Volume, Volatility, Price/Cycle transforms,
    plus some advanced indicators from pandas_ta.
    """

    ########################################################################
    #                         HYPEROPT PARAMETERS
    ########################################################################
    can_short = True

    timeframe = CategoricalParameter(
        ["5m", "15m", "1h"], default="5m", space="buy", optimize=True
    )
    leverage_level = IntParameter(1, 20, default=1, space="buy", optimize=True)

    # Base thresholds
    bullish_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True)
    bearish_threshold = IntParameter(-20, -5, default=-10, space="sell", optimize=True)

    # Stop-loss parameter
    stoploss_param = DecimalParameter(
        -0.25, -0.02, default=-0.05, decimals=2, space="sell", optimize=True
    )

    # Trailing stop-loss parameters
    trailing_stop_param = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=True
    )
    trailing_stop_positive_param = DecimalParameter(
        0.001, 0.05, default=0.02, decimals=3, space="sell", optimize=True
    )
    trailing_stop_positive_offset_param = DecimalParameter(
        0.001, 0.10, default=0.04, decimals=3, space="sell", optimize=True
    )
    trailing_only_offset_is_reached_param = CategoricalParameter(
        [True, False], default=True, space="sell", optimize=True
    )

    # Existing hyperoptable indicators
    ema_short_period = IntParameter(10, 100, default=50, space="buy", optimize=True)
    ema_long_period = IntParameter(100, 300, default=200, space="buy", optimize=True)
    rsi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    macd_fastperiod = IntParameter(8, 16, default=12, space="buy", optimize=True)
    macd_slowperiod = IntParameter(17, 34, default=26, space="buy", optimize=True)
    macd_signalperiod = IntParameter(5, 15, default=9, space="buy", optimize=True)
    bb_window = IntParameter(10, 30, default=20, space="buy", optimize=True)
    bb_stds = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy", optimize=True)

    # Additional hyperoptable indicator periods
    stoch_rsi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    mfi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    cci_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    adx_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    atr_period = IntParameter(10, 30, default=14, space="buy", optimize=True)

    # New Overlap / Momentum / Volume / Volatility / Price-Transform parameters
    dema_period = IntParameter(5, 50, default=20, space="buy", optimize=True)
    tema_period = IntParameter(5, 50, default=20, space="buy", optimize=True)
    sar_acceleration = DecimalParameter(0.001, 0.02, default=0.02, decimals=3, space="buy", optimize=True)
    sar_maximum = DecimalParameter(0.02, 0.5, default=0.2, decimals=2, space="buy", optimize=True)

    stoch_k_period = IntParameter(5, 20, default=14, space="buy", optimize=True)
    stoch_d_period = IntParameter(3, 10, default=3, space="buy", optimize=True)

    willr_period = IntParameter(7, 28, default=14, space="buy", optimize=True)
    roc_period = IntParameter(5, 30, default=10, space="buy", optimize=True)
    ao_fast = IntParameter(5, 14, default=5, space="buy", optimize=True)
    ao_slow = IntParameter(15, 34, default=34, space="buy", optimize=True)

    adl_period = IntParameter(5, 30, default=14, space="buy", optimize=True)  # for smoothing if needed
    cmf_period = IntParameter(5, 30, default=20, space="buy", optimize=True)

    bbw_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    kc_period = IntParameter(10, 30, default=20, space="buy", optimize=True)   # Keltner Channels

    # Price Transform / Cycle Indicators (Hilbert Transform, etc.)
    ht_period = IntParameter(6, 20, default=10, space="buy", optimize=True)

    # Some advanced indicators from pandas_ta:
    supertrend_factor = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space="buy", optimize=True)
    ichimoku_tenkan = IntParameter(7, 12, default=9, space="buy", optimize=True)
    ichimoku_kijun = IntParameter(15, 30, default=26, space="buy", optimize=True)
    donchian_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    kama_period = IntParameter(5, 30, default=10, space="buy", optimize=True)
    crsi_period = IntParameter(5, 20, default=14, space="buy", optimize=True)
    zscore_length = IntParameter(5, 30, default=14, space="buy", optimize=True)

    ########################################################################
    #                         INIT (Constructor)
    ########################################################################
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._leverage_level = int(self.leverage_level.value)
        self._timeframe = str(self.timeframe.value)
        self.stoploss = float(self.stoploss_param.value)

        self.trailing_stop = bool(self.trailing_stop_param.value)
        self.trailing_stop_positive = float(self.trailing_stop_positive_param.value)
        self.trailing_stop_positive_offset = float(self.trailing_stop_positive_offset_param.value)
        self.trailing_only_offset_is_reached = bool(self.trailing_only_offset_is_reached_param.value)

    ########################################################################
    #                         INDICATORS
    ########################################################################
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Computes candlestick patterns, technical indicators, and final 'score'.
        """

        # --- Candlestick Patterns ---
        dataframe = self.calculate_candlestick_patterns(dataframe)

        # --- Core / Existing Indicators (EMA, RSI, MACD, BB, OBV) ---
        short_ema_period = int(self.ema_short_period.value)
        long_ema_period = int(self.ema_long_period.value)
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=short_ema_period)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=long_ema_period)

        rsi_p = int(self.rsi_period.value)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=rsi_p)

        macd_fast = int(self.macd_fastperiod.value)
        macd_slow = int(self.macd_slowperiod.value)
        macd_signal = int(self.macd_signalperiod.value)
        macd = ta.MACD(
            dataframe,
            fastperiod=macd_fast,
            slowperiod=macd_slow,
            signalperiod=macd_signal,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        bb_win = int(self.bb_window.value)
        bb_std = float(self.bb_stds.value)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=bb_win,
            stds=bb_std,
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["obv"] = ta.OBV(dataframe["close"], dataframe["volume"])

        # --- Additional Indicators (STOCH RSI, MFI, CCI, ADX, ATR) ---
        stoch_period = int(self.stoch_rsi_period.value)
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=stoch_period)
        dataframe["stochrsi_k"] = stoch_rsi["fastk"]
        dataframe["stochrsi_d"] = stoch_rsi["fastd"]

        mfi_p = int(self.mfi_period.value)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=mfi_p)

        cci_p = int(self.cci_period.value)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=cci_p)

        adx_p = int(self.adx_period.value)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=adx_p)

        atr_p = int(self.atr_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_p)

        # ------------------------------------------------------------------
        #                  NEWLY-ADDED INDICATORS
        # ------------------------------------------------------------------
        d_period = int(self.dema_period.value)
        dataframe["dema"] = ta.DEMA(dataframe, timeperiod=d_period)

        t_period = int(self.tema_period.value)
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=t_period)

        sar_acc = float(self.sar_acceleration.value)
        sar_max = float(self.sar_maximum.value)
        dataframe["sar"] = ta.SAR(dataframe, acceleration=sar_acc, maximum=sar_max)

        # Simple pivot approximation
        dataframe["pivot"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0

        # Standard Stoch
        stoch_kp = int(self.stoch_k_period.value)
        stoch_dp = int(self.stoch_d_period.value)
        stoch = ta.STOCH(dataframe, fastk_period=stoch_kp, slowk_period=stoch_dp, slowd_period=stoch_dp)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        w_p = int(self.willr_period.value)
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=w_p)

        roc_p = int(self.roc_period.value)
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=roc_p)

        # Awesome Oscillator (AO): difference of two SMAs
        ao_fast_p = int(self.ao_fast.value)
        ao_slow_p = int(self.ao_slow.value)
        sma_fast = ta.SMA(dataframe, timeperiod=ao_fast_p)
        sma_slow = ta.SMA(dataframe, timeperiod=ao_slow_p)
        dataframe["ao"] = sma_fast - sma_slow

        # Volume Indicators: ADL, CMF
        dataframe["adl"] = ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])) / (dataframe["high"] - dataframe["low"] + 1e-10) * dataframe["volume"]
        adl_smooth = int(self.adl_period.value)
        dataframe["adl_smooth"] = ta.EMA(dataframe["adl"], timeperiod=adl_smooth)

        cmf_p = int(self.cmf_period.value)
        # Calculate CMF using pandas_ta
        dataframe['cmf'] = pta.cmf(
            high=dataframe['high'], 
            low=dataframe['low'], 
            close=dataframe['close'], 
            volume=dataframe['volume'], 
            length=cmf_p
        )
        # If using pandas_ta, you can do: dataframe["cmf"] = pta.cmf(...)

        # Volatility Indicators: BBW, Keltner Channels
        bbw_p = int(self.bbw_period.value)
        bb2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=bbw_p, stds=2)
        dataframe["bbw"] = (bb2["upper"] - bb2["lower"]) / bb2["mid"]

        kc_p = int(self.kc_period.value)
        atr_kc = ta.ATR(dataframe, timeperiod=kc_p)
        ma_kc = ta.EMA(dataframe, timeperiod=kc_p)
        dataframe["kc_upper"] = ma_kc + 2.0 * atr_kc
        dataframe["kc_lower"] = ma_kc - 2.0 * atr_kc

        # Hilbert Transform (e.g. Sine Wave)
        ht_p = int(self.ht_period.value)
        try:
            ht_sine = ta.HT_SINE(dataframe)
            dataframe["ht_sine"] = ht_sine["sine"]
            dataframe["ht_leadsine"] = ht_sine["leadsine"]
        except Exception:
            dataframe["ht_sine"] = 0
            dataframe["ht_leadsine"] = 0

        # If using pandas_ta for supertrend, ichimoku, etc., do it here:
        # supertrend_factor_val = float(self.supertrend_factor.value)
        # dataframe["supertrend"] = pta.supertrend(...)

        # Weighted Indicator Scoring
        dataframe["indicator_score"] = 0.0
        dataframe["indicator_trigger"] = ""
        self.apply_indicator_scoring(dataframe)

        return dataframe

    def apply_indicator_scoring(self, dataframe: DataFrame) -> None:
        """
        Assigns weighted scores to each indicator signal, appended to
        'indicator_score' and logs the triggers in 'indicator_trigger'.
        Extend or modify the placeholder logic to match your desired signals.
        """

        # ----------------------------------------------------------------
        # Indicator Weights: Modify as needed for your custom strategy
        # ----------------------------------------------------------------
        indicator_weights = {
            # Existing examples
            "ema_cross_bull": 2.0,
            "ema_cross_bear": -2.0,
            "macd_bull": 2.0,
            "macd_bear": -2.0,
            "rsi_overbought": -3.0,
            "rsi_oversold": 3.0,
            "rsi_midrange_bull": 1.0,
            "rsi_midrange_bear": -1.0,
            "stochrsi_bull": 2.0,
            "stochrsi_bear": -2.0,
            "mfi_overbought": -2.0,
            "mfi_oversold": 2.0,
            "cci_bull": 1.0,
            "cci_bear": -1.0,
            "adx_strong_trend": 1.0,

            # Newly integrated indicators (examples only)
            "dema_bull": 1.5,
            "dema_bear": -1.5,
            "tema_bull": 1.5,
            "tema_bear": -1.5,
            "sar_bull": 2.0,
            "sar_bear": -2.0,
            "pivot_bull": 1.0,
            "pivot_bear": -1.0,
            "stoch_kd_bull": 2.0,
            "stoch_kd_bear": -2.0,
            "willr_overbought": -2.0,
            "willr_oversold": 2.0,
            "roc_bull": 1.0,
            "roc_bear": -1.0,
            "ao_bull": 1.5,
            "ao_bear": -1.5,
            "adl_bull": 1.0,
            "adl_bear": -1.0,
            "cmf_bull": 1.5,
            "cmf_bear": -1.5,
            "bbw_narrow": 1.0,
            "bbw_wide": -1.0,
            "kc_bull": 1.5,
            "kc_bear": -1.5,
            "ht_sine_bull": 1.0,
            "ht_sine_bear": -1.0,
            "supertrend_bull": 2.0,
            "supertrend_bear": -2.0,
            "ichimoku_bull": 2.0,
            "ichimoku_bear": -2.0,
            "donchian_breakout_bull": 2.0,
            "donchian_breakout_bear": -2.0,
            "kama_bull": 1.0,
            "kama_bear": -1.0,
            "crsi_overbought": -2.0,
            "crsi_oversold": 2.0,
            "zscore_above": -1.0,
            "zscore_below": 1.0,
        }

        # -----------------------------------------
        # Example logic for existing indicators:
        # -----------------------------------------
        ema_bull_mask = dataframe["ema_short"] > dataframe["ema_long"]
        ema_bear_mask = dataframe["ema_short"] < dataframe["ema_long"]
        dataframe.loc[ema_bull_mask, "indicator_score"] += indicator_weights["ema_cross_bull"]
        dataframe.loc[ema_bull_mask, "indicator_trigger"] += "EMA_Cross_BULL; "
        dataframe.loc[ema_bear_mask, "indicator_score"] += indicator_weights["ema_cross_bear"]
        dataframe.loc[ema_bear_mask, "indicator_trigger"] += "EMA_Cross_BEAR; "

        macd_bull_mask = dataframe["macd"] > dataframe["macdsignal"]
        macd_bear_mask = dataframe["macd"] < dataframe["macdsignal"]
        dataframe.loc[macd_bull_mask, "indicator_score"] += indicator_weights["macd_bull"]
        dataframe.loc[macd_bull_mask, "indicator_trigger"] += "MACD_BULL; "
        dataframe.loc[macd_bear_mask, "indicator_score"] += indicator_weights["macd_bear"]
        dataframe.loc[macd_bear_mask, "indicator_trigger"] += "MACD_BEAR; "

        # RSI
        rsi = dataframe["rsi"]
        overbought_level = 70
        oversold_level = 30
        rsi_overbought_mask = rsi >= overbought_level
        rsi_oversold_mask = rsi <= oversold_level
        rsi_midrange_bull_mask = (rsi > oversold_level) & (rsi < 50)
        rsi_midrange_bear_mask = (rsi > 50) & (rsi < overbought_level)

        dataframe.loc[rsi_overbought_mask, "indicator_score"] += indicator_weights["rsi_overbought"]
        dataframe.loc[rsi_overbought_mask, "indicator_trigger"] += "RSI_Overbought; "
        dataframe.loc[rsi_oversold_mask, "indicator_score"] += indicator_weights["rsi_oversold"]
        dataframe.loc[rsi_oversold_mask, "indicator_trigger"] += "RSI_Oversold; "
        dataframe.loc[rsi_midrange_bull_mask, "indicator_score"] += indicator_weights["rsi_midrange_bull"]
        dataframe.loc[rsi_midrange_bull_mask, "indicator_trigger"] += "RSI_Mid_BULL; "
        dataframe.loc[rsi_midrange_bear_mask, "indicator_score"] += indicator_weights["rsi_midrange_bear"]
        dataframe.loc[rsi_midrange_bear_mask, "indicator_trigger"] += "RSI_Mid_BEAR; "

        # StochRSI
        stoch_k = dataframe["stochrsi_k"]
        stoch_d = dataframe["stochrsi_d"]
        stoch_bull_mask = stoch_k > stoch_d
        stoch_bear_mask = stoch_k < stoch_d
        dataframe.loc[stoch_bull_mask, "indicator_score"] += indicator_weights["stochrsi_bull"]
        dataframe.loc[stoch_bull_mask, "indicator_trigger"] += "StochRSI_BULL; "
        dataframe.loc[stoch_bear_mask, "indicator_score"] += indicator_weights["stochrsi_bear"]
        dataframe.loc[stoch_bear_mask, "indicator_trigger"] += "StochRSI_BEAR; "

        # MFI
        mfi = dataframe["mfi"]
        mfi_overbought_mask = mfi > 80
        mfi_oversold_mask = mfi < 20
        dataframe.loc[mfi_overbought_mask, "indicator_score"] += indicator_weights["mfi_overbought"]
        dataframe.loc[mfi_overbought_mask, "indicator_trigger"] += "MFI_Overbought; "
        dataframe.loc[mfi_oversold_mask, "indicator_score"] += indicator_weights["mfi_oversold"]
        dataframe.loc[mfi_oversold_mask, "indicator_trigger"] += "MFI_Oversold; "

        # CCI
        cci = dataframe["cci"]
        cci_bull_mask = cci > 100
        cci_bear_mask = cci < -100
        dataframe.loc[cci_bull_mask, "indicator_score"] += indicator_weights["cci_bull"]
        dataframe.loc[cci_bull_mask, "indicator_trigger"] += "CCI_BULL; "
        dataframe.loc[cci_bear_mask, "indicator_score"] += indicator_weights["cci_bear"]
        dataframe.loc[cci_bear_mask, "indicator_trigger"] += "CCI_BEAR; "

        # ADX
        adx_strong_mask = dataframe["adx"] > 25
        dataframe.loc[adx_strong_mask, "indicator_score"] += indicator_weights["adx_strong_trend"]
        dataframe.loc[adx_strong_mask, "indicator_trigger"] += "ADX_StrongTrend; "

        # ---------------------------------------------------------------
        # ADD EXAMPLE LOGIC FOR NEW INDICATORS HERE
        # (Placeholder examples shown below; adapt as needed)
        # ---------------------------------------------------------------

        # DEMA / TEMA crosses (placeholder examples)
        demabull_mask = dataframe["dema"] > dataframe["ema_long"]   # e.g., if DEMA crosses above some baseline
        dataframe.loc[demabull_mask, "indicator_score"] += indicator_weights["dema_bull"]
        dataframe.loc[demabull_mask, "indicator_trigger"] += "DEMA_BULL; "

        demabear_mask = dataframe["dema"] < dataframe["ema_long"]
        dataframe.loc[demabear_mask, "indicator_score"] += indicator_weights["dema_bear"]
        dataframe.loc[demabear_mask, "indicator_trigger"] += "DEMA_BEAR; "

        temabull_mask = dataframe["tema"] > dataframe["ema_long"]
        dataframe.loc[temabull_mask, "indicator_score"] += indicator_weights["tema_bull"]
        dataframe.loc[temabull_mask, "indicator_trigger"] += "TEMA_BULL; "

        temabear_mask = dataframe["tema"] < dataframe["ema_long"]
        dataframe.loc[temabear_mask, "indicator_score"] += indicator_weights["tema_bear"]
        dataframe.loc[temabear_mask, "indicator_trigger"] += "TEMA_BEAR; "

        # SAR flip (placeholder example):
        sar_bull_mask = dataframe["sar"] < dataframe["close"]
        dataframe.loc[sar_bull_mask, "indicator_score"] += indicator_weights["sar_bull"]
        dataframe.loc[sar_bull_mask, "indicator_trigger"] += "SAR_BULL; "

        sar_bear_mask = dataframe["sar"] > dataframe["close"]
        dataframe.loc[sar_bear_mask, "indicator_score"] += indicator_weights["sar_bear"]
        dataframe.loc[sar_bear_mask, "indicator_trigger"] += "SAR_BEAR; "

        # Stoch K/D crossover
        stoch_kd_bull_mask = dataframe["stoch_k"] > dataframe["stoch_d"]
        stoch_kd_bear_mask = dataframe["stoch_k"] < dataframe["stoch_d"]
        dataframe.loc[stoch_kd_bull_mask, "indicator_score"] += indicator_weights["stoch_kd_bull"]
        dataframe.loc[stoch_kd_bull_mask, "indicator_trigger"] += "STOCH_KD_BULL; "
        dataframe.loc[stoch_kd_bear_mask, "indicator_score"] += indicator_weights["stoch_kd_bear"]
        dataframe.loc[stoch_kd_bear_mask, "indicator_trigger"] += "STOCH_KD_BEAR; "

        # WILLR (placeholder: treat below -80 as oversold, above -20 as overbought)
        willr = dataframe["willr"]
        willr_overbought_mask = willr > -20
        willr_oversold_mask = willr < -80
        dataframe.loc[willr_overbought_mask, "indicator_score"] += indicator_weights["willr_overbought"]
        dataframe.loc[willr_overbought_mask, "indicator_trigger"] += "WILLR_Overbought; "
        dataframe.loc[willr_oversold_mask, "indicator_score"] += indicator_weights["willr_oversold"]
        dataframe.loc[willr_oversold_mask, "indicator_trigger"] += "WILLR_Oversold; "

        # ROC (placeholder: if ROC > 0 = bullish momentum, < 0 = bearish)
        roc_bull_mask = dataframe["roc"] > 0
        roc_bear_mask = dataframe["roc"] < 0
        dataframe.loc[roc_bull_mask, "indicator_score"] += indicator_weights["roc_bull"]
        dataframe.loc[roc_bull_mask, "indicator_trigger"] += "ROC_BULL; "
        dataframe.loc[roc_bear_mask, "indicator_score"] += indicator_weights["roc_bear"]
        dataframe.loc[roc_bear_mask, "indicator_trigger"] += "ROC_BEAR; "

        # AO (placeholder: > 0 = bull, < 0 = bear)
        ao_bull_mask = dataframe["ao"] > 0
        ao_bear_mask = dataframe["ao"] < 0
        dataframe.loc[ao_bull_mask, "indicator_score"] += indicator_weights["ao_bull"]
        dataframe.loc[ao_bull_mask, "indicator_trigger"] += "AO_BULL; "
        dataframe.loc[ao_bear_mask, "indicator_score"] += indicator_weights["ao_bear"]
        dataframe.loc[ao_bear_mask, "indicator_trigger"] += "AO_BEAR; "

        # CMF (placeholder: > 0 = bull, < 0 = bear)
        cmf = dataframe["cmf"]
        cmf_bull_mask = cmf > 0
        cmf_bear_mask = cmf < 0
        dataframe.loc[cmf_bull_mask, "indicator_score"] += indicator_weights["cmf_bull"]
        dataframe.loc[cmf_bull_mask, "indicator_trigger"] += "CMF_BULL; "
        dataframe.loc[cmf_bear_mask, "indicator_score"] += indicator_weights["cmf_bear"]
        dataframe.loc[cmf_bear_mask, "indicator_trigger"] += "CMF_BEAR; "

        # BBW (placeholder: define “narrow” vs. “wide” according to your strategy)
        bbw = dataframe["bbw"]
        bbw_narrow_mask = bbw < 0.05  # Example threshold
        bbw_wide_mask = bbw > 0.15   # Example threshold
        dataframe.loc[bbw_narrow_mask, "indicator_score"] += indicator_weights["bbw_narrow"]
        dataframe.loc[bbw_narrow_mask, "indicator_trigger"] += "BBW_NARROW; "
        dataframe.loc[bbw_wide_mask, "indicator_score"] += indicator_weights["bbw_wide"]
        dataframe.loc[bbw_wide_mask, "indicator_trigger"] += "BBW_WIDE; "

        # Keltner Channels (placeholder: if close near upper band = bull, near lower = bear)
        kc_bull_mask = dataframe["close"] > dataframe["kc_upper"]
        kc_bear_mask = dataframe["close"] < dataframe["kc_lower"]
        dataframe.loc[kc_bull_mask, "indicator_score"] += indicator_weights["kc_bull"]
        dataframe.loc[kc_bull_mask, "indicator_trigger"] += "KC_BULL; "
        dataframe.loc[kc_bear_mask, "indicator_score"] += indicator_weights["kc_bear"]
        dataframe.loc[kc_bear_mask, "indicator_trigger"] += "KC_BEAR; "

        # HT SINE (placeholder: if sine > leadsine => bull, etc.)
        ht_bull_mask = dataframe["ht_sine"] > dataframe["ht_leadsine"]
        ht_bear_mask = dataframe["ht_sine"] < dataframe["ht_leadsine"]
        dataframe.loc[ht_bull_mask, "indicator_score"] += indicator_weights["ht_sine_bull"]
        dataframe.loc[ht_bull_mask, "indicator_trigger"] += "HT_SINE_BULL; "
        dataframe.loc[ht_bear_mask, "indicator_score"] += indicator_weights["ht_sine_bear"]
        dataframe.loc[ht_bear_mask, "indicator_trigger"] += "HT_SINE_BEAR; "

        # Supertrend, Ichimoku, Donchian, KAMA, CRSI, Z-score, etc.
        # Placeholder logic. The actual signals must be defined:
        # e.g., if supertrend is in 'uptrend', add bull, else bear

        # PLEASE ADD REAL DETECTION LOGIC FOR:
        #   'supertrend_bull', 'supertrend_bear',
        #   'ichimoku_bull', 'ichimoku_bear',
        #   'donchian_breakout_bull', 'donchian_breakout_bear',
        #   'kama_bull', 'kama_bear',
        #   'crsi_overbought', 'crsi_oversold',
        #   'zscore_above', 'zscore_below'
        # For example:
        # supertrend_bull_mask = (some_condition_for_supertrend_uptrend)
        # dataframe.loc[supertrend_bull_mask, "indicator_score"] += indicator_weights["supertrend_bull"]
        # dataframe.loc[supertrend_bull_mask, "indicator_trigger"] += "SUPER_BULL; "

        # etc.

    ########################################################################
    #                  ALL TA-LIB CANDLESTICK PATTERNS
    ########################################################################
    def calculate_candlestick_patterns(self, df: DataFrame) -> DataFrame:
        """
        Identifies all candlestick patterns provided by TA-Lib and assigns a
        bullish or bearish score accordingly. The final sum is stored in
        'candlestick_score'.
        """
        all_patterns = {
            "CDL2CROWS": ta.CDL2CROWS,
            "CDL3BLACKCROWS": ta.CDL3BLACKCROWS,
            "CDL3INSIDE": ta.CDL3INSIDE,
            "CDL3LINESTRIKE": ta.CDL3LINESTRIKE,
            "CDL3OUTSIDE": ta.CDL3OUTSIDE,
            "CDL3STARSINSOUTH": ta.CDL3STARSINSOUTH,
            "CDL3WHITESOLDIERS": ta.CDL3WHITESOLDIERS,
            "CDLABANDONEDBABY": ta.CDLABANDONEDBABY,
            "CDLADVANCEBLOCK": ta.CDLADVANCEBLOCK,
            "CDLBELTHOLD": ta.CDLBELTHOLD,
            "CDLBREAKAWAY": ta.CDLBREAKAWAY,
            "CDLCLOSINGMARUBOZU": ta.CDLCLOSINGMARUBOZU,
            "CDLCONCEALBABYSWALL": ta.CDLCONCEALBABYSWALL,
            "CDLCOUNTERATTACK": ta.CDLCOUNTERATTACK,
            "CDLDARKCLOUDCOVER": ta.CDLDARKCLOUDCOVER,
            "CDLDOJI": ta.CDLDOJI,
            "CDLDOJISTAR": ta.CDLDOJISTAR,
            "CDLDRAGONFLYDOJI": ta.CDLDRAGONFLYDOJI,
            "CDLENGULFING": ta.CDLENGULFING,
            "CDLEVENINGDOJISTAR": ta.CDLEVENINGDOJISTAR,
            "CDLEVENINGSTAR": ta.CDLEVENINGSTAR,
            "CDLGAPSIDESIDEWHITE": ta.CDLGAPSIDESIDEWHITE,
            "CDLGRAVESTONEDOJI": ta.CDLGRAVESTONEDOJI,
            "CDLHAMMER": ta.CDLHAMMER,
            "CDLHANGINGMAN": ta.CDLHANGINGMAN,
            "CDLHARAMI": ta.CDLHARAMI,
            "CDLHARAMICROSS": ta.CDLHARAMICROSS,
            "CDLHIGHWAVE": ta.CDLHIGHWAVE,
            "CDLHIKKAKE": ta.CDLHIKKAKE,
            "CDLHIKKAKEMOD": ta.CDLHIKKAKEMOD,
            "CDLHOMINGPIGEON": ta.CDLHOMINGPIGEON,
            "CDLIDENTICAL3CROWS": ta.CDLIDENTICAL3CROWS,
            "CDLINNECK": ta.CDLINNECK,
            "CDLINVERTEDHAMMER": ta.CDLINVERTEDHAMMER,
            "CDLKICKING": ta.CDLKICKING,
            "CDLKICKINGBYLENGTH": ta.CDLKICKINGBYLENGTH,
            "CDLLADDERBOTTOM": ta.CDLLADDERBOTTOM,
            "CDLLONGLEGGEDDOJI": ta.CDLLONGLEGGEDDOJI,
            "CDLLONGLINE": ta.CDLLONGLINE,
            "CDLMARUBOZU": ta.CDLMARUBOZU,
            "CDLMATCHINGLOW": ta.CDLMATCHINGLOW,
            "CDLMATHOLD": ta.CDLMATHOLD,
            "CDLMORNINGDOJISTAR": ta.CDLMORNINGDOJISTAR,
            "CDLMORNINGSTAR": ta.CDLMORNINGSTAR,
            "CDLONNECK": ta.CDLONNECK,
            "CDLPIERCING": ta.CDLPIERCING,
            "CDLRICKSHAWMAN": ta.CDLRICKSHAWMAN,
            "CDLRISEFALL3METHODS": ta.CDLRISEFALL3METHODS,
            "CDLSEPARATINGLINES": ta.CDLSEPARATINGLINES,
            "CDLSHOOTINGSTAR": ta.CDLSHOOTINGSTAR,
            "CDLSHORTLINE": ta.CDLSHORTLINE,
            "CDLSPINNINGTOP": ta.CDLSPINNINGTOP,
            "CDLSTALLEDPATTERN": ta.CDLSTALLEDPATTERN,
            "CDLSTICKSANDWICH": ta.CDLSTICKSANDWICH,
            "CDLTAKURI": ta.CDLTAKURI,
            "CDLTASUKIGAP": ta.CDLTASUKIGAP,
            "CDLTHRUSTING": ta.CDLTHRUSTING,
            "CDLTRISTAR": ta.CDLTRISTAR,
            "CDLUNIQUE3RIVER": ta.CDLUNIQUE3RIVER,
            "CDLUPSIDEGAP2CROWS": ta.CDLUPSIDEGAP2CROWS,
            "CDLXSIDEGAP3METHODS": ta.CDLXSIDEGAP3METHODS,
        }

        pattern_weights = {
            "CDLENGULFING": 2.0,
            "CDLHAMMER": 2.0,
            "CDLSHOOTINGSTAR": 2.0,
        }

        df["candlestick_score"] = 0.0
        df["pattern_trigger"] = ""

        for pattern_name, pattern_func in all_patterns.items():
            df[pattern_name] = pattern_func(df)
            bullish_mask = df[pattern_name] > 0
            bearish_mask = df[pattern_name] < 0
            weight = pattern_weights.get(pattern_name, 1.0)

            df.loc[bullish_mask, "candlestick_score"] += weight
            df.loc[bullish_mask, "pattern_trigger"] += f"{pattern_name}_BULL; "
            df.loc[bearish_mask, "candlestick_score"] -= weight
            df.loc[bearish_mask, "pattern_trigger"] += f"{pattern_name}_BEAR; "

        return df

    ########################################################################
    #                         ENTRY TREND
    ########################################################################
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enters trades with dynamic thresholds that increase if the market 
        exhibits consecutive bullish or bearish runs. The idea is to 
        require stronger signals when you're trading against an entrenched trend.
        """
        # 1) Combine candlestick + indicator
        dataframe["score"] = dataframe["candlestick_score"] + dataframe["indicator_score"]

        # 2) Track consecutive bullish/bearish streaks
        dataframe["bullish_run"] = 0
        dataframe["bearish_run"] = 0

        bullish_count = 0
        bearish_count = 0

        for idx in range(len(dataframe)):
            sc = dataframe.at[idx, "score"]
            if sc > 0:
                bullish_count += 1
                bearish_count = 0
            elif sc < 0:
                bearish_count += 1
                bullish_count = 0
            else:
                bullish_count = 0
                bearish_count = 0

            dataframe.at[idx, "bullish_run"] = bullish_count
            dataframe.at[idx, "bearish_run"] = bearish_count

        # 3) Dynamic thresholds
        base_bull_th = float(self.bullish_threshold.value)
        base_bear_th = float(self.bearish_threshold.value)

        dataframe["final_bullish_th"] = base_bull_th + (dataframe["bearish_run"] * 0.5)
        dataframe["final_bearish_th"] = base_bear_th - (dataframe["bullish_run"] * 0.5)

        # 4) Evaluate dynamic conditions for entry
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["entry_pattern"] = ""
        dataframe["entry_indicator"] = ""

        bull_condition = dataframe["score"] >= dataframe["final_bullish_th"]
        bear_condition = dataframe["score"] <= dataframe["final_bearish_th"]

        dataframe.loc[bull_condition, "enter_long"] = 1
        dataframe.loc[bull_condition, "entry_pattern"] = dataframe["pattern_trigger"]
        dataframe.loc[bull_condition, "entry_indicator"] = dataframe["indicator_trigger"]

        dataframe.loc[bear_condition, "enter_short"] = 1
        dataframe.loc[bear_condition, "entry_pattern"] = dataframe["pattern_trigger"]
        dataframe.loc[bear_condition, "entry_indicator"] = dataframe["indicator_trigger"]

        # Optional: Keep signals active for 50 candles
        dataframe["enter_long"] = dataframe["enter_long"].rolling(50, min_periods=1).max()
        dataframe["enter_short"] = dataframe["enter_short"].rolling(50, min_periods=1).max()

        return dataframe

    ########################################################################
    #                         EXIT TREND
    ########################################################################
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Default exit logic is empty, relying on stop-loss or trailing stop.
        You can add exit conditions if desired.
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    ########################################################################
    #                         LEVERAGE HOOK
    ########################################################################
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs
    ) -> float:
        return float(self._leverage_level)
