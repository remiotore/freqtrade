"""
Freqtrade strategy equivalent to the TradingView Pine Script (BinHV27_combined).
Replicates the same parameters, indicators, entry/exit logic, dynamic stop-loss, etc.

You must enable "can_short = True" in order to allow short entries, and configure
Freqtrade for margin or futures accordingly.

Disclaimer: The code is provided as an illustrative example; you may need to
adjust it to your environment or for minor syntax changes if using newer versions
of freqtrade/pandas_ta.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Freqtrade imports
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IntParameter, DecimalParameter, BooleanParameter)
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes

# Third-party libraries
import talib.abstract as ta

logger = logging.getLogger(__name__)

#################################################################
# Helper functions
#################################################################

def linreg(series: pd.Series, period: int = 2) -> pd.Series:
    """
    Mimics TradingView's ta.linreg(series, period=2, offset=0).
    In TradingView, linreg(...) = linear regression of 'period' bars with 0 offset.
    For an exact TSF/linreg(2,0), the slope typically uses 1…period sample.
    The formula for TSF (Time Series Forecast) at bar i is:
        TSF(i) = a0 + a1 * (period-1)
    where a1 is slope, a0 intercept.
    
    A simpler approach for small periods:
      - We can use a direct slope calculation between current bar and bar N bars ago
        or
      - Use a standard least-squares polynomial fit on the last `period` points.
    
    Because period=2 is very short, the slope is basically (series - series.shift(1)).
    We'll do a minimal approach that tries to replicate tv's linreg(…,2).
    """
    # For period=2 specifically, it's basically last 2 bars slope-based forecast:
    # slope = (val1 - val0)/1, intercept = val1 - slope*1
    # TSF(current) = intercept + slope*(period-1) = intercept + slope
    # which effectively equals val1 for offset=0. 
    # However, to emulate typical "linreg" with a 'least squares' approach for period=2:
    
    # Using a standard formula for "least squares" on 2 bars is almost the same as the last known price.
    # We'll implement a generic approach for any period using polyfit:
    linreg_vals = series.rolling(period).apply(
        lambda x: np.polyval(np.polyfit(range(period), x.values, 1), period-1),
        raw=False
    )
    return linreg_vals


def hlc3(df: pd.DataFrame) -> pd.Series:
    """Return typical price (H+L+C)/3 for each row."""
    return (df['high'] + df['low'] + df['close']) / 3.0


#################################################################
# Main Strategy Class
#################################################################

class BinHV27CombinedStrategy(IStrategy):
    """
    Converted from Pine Script:
    - Strategy name: "BinHV27_combined"
    - Emulates all parameters, logic, dynamic stoploss, custom exit.
    """

    ####################################################
    # --- Standard Freqtrade parameters
    ####################################################
    INTERFACE_VERSION = 3

    # This example uses 1h as base timeframe. 
    # Adjust if your Pine Script used a different base resolution.
    timeframe = '1h'

    # For multi-timeframe analysis, we fetch 4h data via informative pair.
    informative_timeframe = '4h'

    # Enable short for margin/futures mode. 
    can_short = True

    # Minimal ROI and stoploss just as placeholders. We'll override with custom stoploss.
    minimal_roi = {"0": 100}  # effectively no static ROI exit
    stoploss = -1  # we rely on custom_stoploss.

    # If you want to allow custom exit, set to True
    use_custom_stoploss = True
    use_custom_exit = True

    ####################################################
    # --- Pine Script Inputs as Freqtrade parameters ---
    ####################################################

    # 1) Buy parameters (long)
    entry_long_adx1 = IntParameter(10, 100, default=25, space="buy", optimize=False)
    entry_long_emarsi1 = IntParameter(10, 100, default=20, space="buy", optimize=False)
    entry_long_adx2 = IntParameter(20, 100, default=30, space="buy", optimize=False)
    entry_long_emarsi2 = IntParameter(20, 100, default=20, space="buy", optimize=False)
    entry_long_adx3 = IntParameter(10, 100, default=35, space="buy", optimize=False)
    entry_long_emarsi3 = IntParameter(10, 100, default=20, space="buy", optimize=False)
    entry_long_adx4 = IntParameter(20, 100, default=30, space="buy", optimize=False)
    entry_long_emarsi4 = IntParameter(20, 100, default=25, space="buy", optimize=False)

    # 2) Buy parameters (short)
    entry_short_adx1 = IntParameter(10, 100, default=62, space="buy", optimize=False)
    entry_short_emarsi1 = IntParameter(10, 100, default=29, space="buy", optimize=False)
    entry_short_adx2 = IntParameter(20, 100, default=29, space="buy", optimize=False)
    entry_short_emarsi2 = IntParameter(20, 100, default=30, space="buy", optimize=False)
    entry_short_adx3 = IntParameter(10, 100, default=33, space="buy", optimize=False)
    entry_short_emarsi3 = IntParameter(10, 100, default=22, space="buy", optimize=False)
    entry_short_adx4 = IntParameter(20, 100, default=88, space="buy", optimize=False)
    entry_short_emarsi4 = IntParameter(20, 100, default=57, space="buy", optimize=False)

    # 3) Dynamic stop parameters (long)
    pHSL_long = DecimalParameter(-0.99, -0.04, default=-0.25, space="sell", optimize=False)
    pPF_1_long = DecimalParameter(0.008, 0.1, default=0.012, space="sell", optimize=False)
    pSL_1_long = DecimalParameter(0.008, 0.1, default=0.01, space="sell", optimize=False)
    pPF_2_long = DecimalParameter(0.04, 0.2, default=0.05, space="sell", optimize=False)
    pSL_2_long = DecimalParameter(0.04, 0.2, default=0.04, space="sell", optimize=False)

    # 4) Dynamic stop parameters (short)
    pHSL_short = DecimalParameter(-0.99, -0.04, default=-0.863, space="sell", optimize=False)
    pPF_1_short = DecimalParameter(0.008, 0.1, default=0.018, space="sell", optimize=False)
    pSL_1_short = DecimalParameter(0.008, 0.1, default=0.015, space="sell", optimize=False)
    pPF_2_short = DecimalParameter(0.04, 0.2, default=0.197, space="sell", optimize=False)
    pSL_2_short = DecimalParameter(0.04, 0.2, default=0.157, space="sell", optimize=False)

    # 5) Exit parameters (long)
    exit_long_emarsi1 = IntParameter(10, 100, default=75, space="sell", optimize=False)
    exit_long_adx2    = IntParameter(10, 100, default=30, space="sell", optimize=False)
    exit_long_emarsi2 = IntParameter(20, 100, default=80, space="sell", optimize=False)
    exit_long_emarsi3 = IntParameter(20, 100, default=75, space="sell", optimize=False)

    # 6) Exit parameters (short)
    exit_short_emarsi1 = IntParameter(10, 100, default=30, space="sell", optimize=False)
    exit_short_adx2    = IntParameter(10, 100, default=21, space="sell", optimize=False)
    exit_short_emarsi2 = IntParameter(20, 100, default=71, space="sell", optimize=False)
    exit_short_emarsi3 = IntParameter(20, 100, default=72, space="sell", optimize=False)

    # 7) Exit switches
    exit_long_1  = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_long_2  = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_long_3  = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_long_4  = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_long_5  = BooleanParameter(default=True,  space="sell", optimize=False)

    exit_short_1 = BooleanParameter(default=False, space="sell", optimize=False)
    exit_short_2 = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_short_3 = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_short_4 = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_short_5 = BooleanParameter(default=False, space="sell", optimize=False)

    # 8) Leverage param (informational only)
    leverage_num = IntParameter(1, 5, default=1, space="buy", optimize=False)

    ###############################################################
    # Informative pairs: We fetch 4h data for multi-timeframe logic
    ###############################################################
    def informative_pairs(self) -> List[tuple]:
        """
        Define additional (pair, timeframe) combinations to fetch for analysis.
        We'll request the same pair at the 4h timeframe.
        """
        pairs = []
        # We want to fetch the base pair at 4h
        pairs.append((self.dp.current_pair, self.informative_timeframe))
        return pairs

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate indicators for the 1h (base) timeframe.
        We'll also merge the 4h indicators (linreg-based) after computing them
        in a separate method or by using the informative pair logic.
        """
        # -------------------------------------------------------------
        # 1) Standard indicators on 1h (base timeframe)
        # -------------------------------------------------------------
        # RSI(5)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=5)
        # EMA of RSI(5)
        dataframe['emarsi'] = ta.EMA(dataframe['rsi'], timeperiod=5)

        # DMI(14) => ADX, plusDI, minusDI
        adx_all = ta.DX(dataframe, timeperiod=14)
        # ta.DX returns ADX alone. We also need +DI, -DI from a custom approach:
        # We'll replicate them quickly:
        # plusDI = 100 * (EMA( Max( (+DM,0) ), 14 ) / ATR(14))
        # minusDI = 100 * (EMA( Max( (-DM,0) ), 14 ) / ATR(14))
        # or we can do a quick approach from pandas_ta. 
        # If you prefer pure TA-Lib, you'd do:
        #   +DI = ta.PLUS_DI(dataframe, 14)
        #   -DI = ta.MINUS_DI(dataframe, 14)
        # but let's do that:

        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['adx'] = adx_all

        # -DIEMA(25) / +DIEMA(5)
        dataframe['minus_di_ema25'] = ta.EMA(dataframe['minus_di'], timeperiod=25)
        dataframe['plus_di_ema5']   = ta.EMA(dataframe['plus_di'],   timeperiod=5)

        # EMA(60), EMA(120), SMA(120), SMA(240)
        dataframe['ema60']  = ta.EMA(dataframe, timeperiod=60)
        dataframe['ema120'] = ta.EMA(dataframe, timeperiod=120)
        dataframe['sma120'] = ta.SMA(dataframe, timeperiod=120)
        dataframe['sma240'] = ta.SMA(dataframe, timeperiod=240)

        # bigup / bigdown logic
        # bigup = (fastsma > slowsma) and ((fastsma - slowsma) > close/300)
        dataframe['fastsma'] = dataframe['sma120']
        dataframe['slowsma'] = dataframe['sma240']
        dataframe['bigup'] = (
            (dataframe['fastsma'] > dataframe['slowsma']) &
            ((dataframe['fastsma'] - dataframe['slowsma']) > (dataframe['close'] / 300))
        )
        dataframe['bigdown'] = ~dataframe['bigup']

        # "trend" for difference and checks
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        # preparechangetrend = trend > trend[1]
        dataframe['trend_prev'] = dataframe['trend'].shift(1)
        dataframe['trend_prev2'] = dataframe['trend'].shift(2)
        dataframe['preparechangetrend'] = dataframe['trend'] > dataframe['trend_prev']
        dataframe['preparechangetrendconfirm'] = (
            dataframe['preparechangetrend'] &
            (dataframe['trend_prev'] > dataframe['trend_prev2'])
        )
        # continueup = (slowsma>slowsma[1]) and (slowsma[1]>slowsma[2])
        dataframe['slowsma_prev'] = dataframe['slowsma'].shift(1)
        dataframe['slowsma_prev2'] = dataframe['slowsma'].shift(2)
        dataframe['continueup'] = (
            (dataframe['slowsma'] > dataframe['slowsma_prev']) &
            (dataframe['slowsma_prev'] > dataframe['slowsma_prev2'])
        )
        # delta = fastsma - fastsma[1]
        dataframe['fastsma_prev'] = dataframe['fastsma'].shift(1)
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma_prev']
        dataframe['delta_prev'] = dataframe['delta'].shift(1)
        dataframe['slowingdown'] = dataframe['delta'] < dataframe['delta_prev']

        # Bollinger (20,2) for reference (not directly used in entry/exit, but keep it)
        bb_basis = ta.SMA(hlc3(dataframe), timeperiod=20)
        bb_std = 2.0 * pd.Series.rolling(hlc3(dataframe), window=20).std()
        dataframe['bb_upperband']  = bb_basis + bb_std
        dataframe['bb_middleband'] = bb_basis
        dataframe['bb_lowerband']  = bb_basis - bb_std

        # -------------------------------------------------------------
        # 2) Fetch and merge 4h informative data
        # -------------------------------------------------------------
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        # We compute h_close4H, h_high4H, h_low4H => then do hlc3_4h => then linreg(2)
        informative['h_close4H'] = informative['close']
        informative['h_high4H']  = informative['high']
        informative['h_low4H']   = informative['low']
        informative['hlc3_4h']   = hlc3(informative)
        # replicate ta.linreg(hlc3_4h, 2, 0)
        informative['tsf_4h'] = linreg(informative['hlc3_4h'], period=2)

        # Only keep the columns we need, and rename them for clarity
        informative = informative[['date', 'hlc3_4h', 'tsf_4h']]

        # Merge with base timeframe
        dataframe = dataframe.merge(
            informative,
            on='date',
            how='left',
            suffixes=('', '_4h')
        )

        # Now the allow_long, allow_short logic:
        # allow_long = (tsf_4h / hlc3_4h) > 1.01
        # allow_short = (tsf_4h / hlc3_4h) < 0.99
        dataframe['allow_long'] = (
            (dataframe['tsf_4h'] / dataframe['hlc3_4h']) > 1.01
        )
        dataframe['allow_short'] = (
            (dataframe['tsf_4h'] / dataframe['hlc3_4h']) < 0.99
        )

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate buy (long) and sell (short) signals.
        Because Freqtrade "buy" = open a long position in spot mode,
        and "sell" = close a long. To handle short, we must do a
        separate approach or rely on custom signals (or "entry_tag" approach).
        However, with "can_short = True", we can define "buy" for short if we set
        a signal = is_short: True. We'll do that below in populate_sell_trend or custom logic.
        """

        # --- LONG ENTRY LOGIC (mirror Pine Script) ---
        # Condition: enterLong = (long_entry_1 OR long_entry_2 OR long_entry_3 OR long_entry_4)

        # For readability, define a few aliases:
        df = dataframe
        # Shorter references to parameters:
        a1 = self.entry_long_adx1.value
        e1 = self.entry_long_emarsi1.value
        a2 = self.entry_long_adx2.value
        e2 = self.entry_long_emarsi2.value
        a3 = self.entry_long_adx3.value
        e3 = self.entry_long_emarsi3.value
        a4 = self.entry_long_adx4.value
        e4 = self.entry_long_emarsi4.value

        # Each sub-condition:
        long_entry_1 = (
            (df['allow_long']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (~df['preparechangetrend']) &
            (~df['continueup']) &
            (df['adx'] > a1) &
            (df['bigdown']) &
            (df['emarsi'] <= e1)
        )

        long_entry_2 = (
            (df['allow_long']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (~df['preparechangetrend']) &
            (df['continueup']) &
            (df['adx'] > a2) &
            (df['bigdown']) &
            (df['emarsi'] <= e2)
        )

        long_entry_3 = (
            (df['allow_long']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (~df['continueup']) &
            (df['adx'] > a3) &
            (df['bigup']) &
            (df['emarsi'] <= e3)
        )

        long_entry_4 = (
            (df['allow_long']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (df['continueup']) &
            (df['adx'] > a4) &
            (df['bigup']) &
            (df['emarsi'] <= e4)
        )

        df.loc[
            (long_entry_1 | long_entry_2 | long_entry_3 | long_entry_4),
            ['buy','buy_tag']
        ] = (1, 'enter_long')

        return df

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        This is used for short entries if `can_short = True`.
        In freqtrade, "sell" can close a long or open a short if "is_short" is set to True.
        We'll place the short-entry logic here by setting `df['sell'] = 1` with `is_short = True`.
        That instructs freqtrade to open a short position.
        However, remember that the standard "sell" also closes a long if there's an open long. 
        Freqtrade’s logic is a bit simpler if you use custom signals (entry_tag approach).
        We'll do the simpler approach: We instruct freqtrade to open short positions by
        returning an 'entry_tag' for short. 
        """

        df = dataframe
        # Shorter references:
        a1 = self.entry_short_adx1.value
        e1 = self.entry_short_emarsi1.value
        a2 = self.entry_short_adx2.value
        e2 = self.entry_short_emarsi2.value
        a3 = self.entry_short_adx3.value
        e3 = self.entry_short_emarsi3.value
        a4 = self.entry_short_adx4.value
        e4 = self.entry_short_emarsi4.value

        short_entry_1 = (
            (df['allow_short']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (~df['preparechangetrend']) &
            (~df['continueup']) &
            (df['adx'] > a1) &
            (df['bigdown']) &
            (df['emarsi'] <= e1)
        )

        short_entry_2 = (
            (df['allow_short']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (~df['preparechangetrend']) &
            (df['continueup']) &
            (df['adx'] > a2) &
            (df['bigdown']) &
            (df['emarsi'] <= e2)
        )

        short_entry_3 = (
            (df['allow_short']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (~df['continueup']) &
            (df['adx'] > a3) &
            (df['bigup']) &
            (df['emarsi'] <= e3)
        )

        short_entry_4 = (
            (df['allow_short']) &
            (df['slowsma'] > 0) &
            (df['close'] < df['ema120']) &
            (df['close'] < df['ema60']) &
            (df['minus_di'] > df['minus_di_ema25']) &
            (df['rsi'] >= df['rsi'].shift(1)) &
            (df['continueup']) &
            (df['adx'] > a4) &
            (df['bigup']) &
            (df['emarsi'] <= e4)
        )

        # Combine short signals:
        short_signal = (short_entry_1 | short_entry_2 | short_entry_3 | short_entry_4)

        # In freqtrade, to open short from "sell" side, we do:
        df.loc[short_signal, ['sell', 'sell_tag', 'is_short']] = (1, 'enter_short', True)

        # Otherwise, do nothing:
        return df

    ######################################################
    # Custom Stoploss to replicate the dynamic stops
    ######################################################
    def custom_stoploss(self, pair: str, trade: Trade, current_time: pd.Timestamp,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Replicates the dynamic stoploss logic from the Pine Script:
          - pHSL
          - pPF_1 / pSL_1
          - pPF_2 / pSL_2
          - linear interpolation if current_profit is between pPF_1 and pPF_2
          - For short, everything is reversed in terms of direction
        Returns the stoploss (e.g. -0.10 for 10% stop).
        If we want no stop (or very large) in certain conditions, return 1 to disable or a big negative.
        """
        # Distinguish if this trade is short or long:
        is_short = trade.is_short

        if not is_short:
            # Long position
            hsl  = self.pHSL_long.value   # Hard stop
            pf_1 = self.pPF_1_long.value
            sl_1 = self.pSL_1_long.value
            pf_2 = self.pPF_2_long.value
            sl_2 = self.pSL_2_long.value

            if current_profit < pf_1:
                return abs(hsl)
            elif current_profit > pf_2:
                # stop = sl_2 + (current_profit - pf_2)
                return sl_2 + (current_profit - pf_2)
            else:
                # linear interpolation
                pct = (current_profit - pf_1) / (pf_2 - pf_1)
                return sl_1 + pct*(sl_2 - sl_1)

        else:
            # Short position
            hsl  = self.pHSL_short.value
            pf_1 = self.pPF_1_short.value
            sl_1 = self.pSL_1_short.value
            pf_2 = self.pPF_2_short.value
            sl_2 = self.pSL_2_short.value

            if current_profit < pf_1:
                return abs(hsl)
            elif current_profit > pf_2:
                return sl_2 + (current_profit - pf_2)
            else:
                pct = (current_profit - pf_1) / (pf_2 - pf_1)
                return sl_1 + pct*(sl_2 - sl_1)

    ######################################################
    # Custom Exit to replicate all “exit_*” conditions
    ######################################################
    def custom_exit(self, pair: str, trade: Trade, current_time: pd.Timestamp,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Checks the additional exit conditions from the Pine Script:
        exit_long_1/2/3/4/5 and exit_short_1/2/3/4/5.  
        
        Return a non-empty string (exit signal name) to close the position.
        Return None to keep the position open.
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        # We need the last row that matches current_time exactly (or the nearest index)
        # Because of possible candle alignment issues, let's do a safer approach:
        # find the candle in 'dataframe' that is <= current_time
        row = dataframe.loc[dataframe['date'] == current_time]
        if row.empty:
            # If no exact match, find last index less than current_time
            row = dataframe.loc[dataframe['date'] < current_time]
            if row.empty:
                return None
            row = row.iloc[-1]  # last available
        else:
            row = row.iloc[-1]

        is_short = trade.is_short

        # Gather booleans from strategy
        preparechangetrendconfirm = bool(row['preparechangetrendconfirm'])
        continueup = bool(row['continueup'])
        slowingdown = bool(row['slowingdown'])
        bigdown = bool(row['bigdown'])
        bigup = bool(row['bigup'])

        # (minusdi < plusdi)
        minus_di = row['minus_di']
        plus_di  = row['plus_di']

        # Also read the parameter toggles
        el1 = self.exit_long_1.value
        el2 = self.exit_long_2.value
        el3 = self.exit_long_3.value
        el4 = self.exit_long_4.value
        el5 = self.exit_long_5.value

        es1 = self.exit_short_1.value
        es2 = self.exit_short_2.value
        es3 = self.exit_short_3.value
        es4 = self.exit_short_4.value
        es5 = self.exit_short_5.value

        # Additional columns we need for conditions:
        close = row['close']
        ema60 = row['ema60']
        ema120 = row['ema120']
        slowsma = row['slowsma']
        adxVal = row['adx']
        emarsi = row['emarsi']

        # -----------
        # Multi-Exit logic for LONG
        # -----------
        if not is_short:
            # "longPos" exit conditions:
            exit_cond_long_1 = (
                el1 and
                (not preparechangetrendconfirm) and
                (not continueup) and
                ((close > ema60) or (close > ema120)) and
                (ema120 > 0) and
                bigdown
            )

            exit_cond_long_2 = (
                el2 and
                (not preparechangetrendconfirm) and
                (not continueup) and
                (close > ema120) and
                (ema120 > 0) and
                ((emarsi > self.exit_long_emarsi1.value) or (close > slowsma)) and
                bigdown
            )

            exit_cond_long_3 = (
                el3 and
                (not preparechangetrendconfirm) and
                (close > ema120) and
                (ema120 > 0) and
                (adxVal > self.exit_long_adx2.value) and
                (emarsi >= self.exit_long_emarsi2.value) and
                bigup
            )

            exit_cond_long_4 = (
                el4 and
                preparechangetrendconfirm and
                (not continueup) and
                slowingdown and
                (emarsi >= self.exit_long_emarsi3.value) and
                (slowsma > 0)
            )

            exit_cond_long_5 = (
                el5 and
                preparechangetrendconfirm and
                (minus_di < plus_di) and
                (close > ema60) and
                (slowsma > 0)
            )

            exit_long = (
                exit_cond_long_1 or
                exit_cond_long_2 or
                exit_cond_long_3 or
                exit_cond_long_4 or
                exit_cond_long_5
            )

            if exit_long:
                return "CustomExitLong"

        else:
            # -----------
            # Multi-Exit logic for SHORT
            # -----------
            exit_cond_short_1 = (
                es1 and
                (not preparechangetrendconfirm) and
                (not continueup) and
                ((close > ema60) or (close > ema120)) and
                (ema120 > 0) and
                bigdown
            )

            exit_cond_short_2 = (
                es2 and
                (not preparechangetrendconfirm) and
                (not continueup) and
                (close > ema120) and
                (ema120 > 0) and
                ((emarsi > self.exit_short_emarsi1.value) or (close > slowsma)) and
                bigdown
            )

            exit_cond_short_3 = (
                es3 and
                (not preparechangetrendconfirm) and
                (close > ema120) and
                (ema120 > 0) and
                (adxVal > self.exit_short_adx2.value) and
                (emarsi >= self.exit_short_emarsi2.value) and
                bigup
            )

            exit_cond_short_4 = (
                es4 and
                preparechangetrendconfirm and
                (not continueup) and
                slowingdown and
                (emarsi >= self.exit_short_emarsi3.value) and
                (slowsma > 0)
            )

            exit_cond_short_5 = (
                es5 and
                preparechangetrendconfirm and
                (minus_di < plus_di) and
                (close > ema60) and
                (slowsma > 0)
            )

            exit_short = (
                exit_cond_short_1 or
                exit_cond_short_2 or
                exit_cond_short_3 or
                exit_cond_short_4 or
                exit_cond_short_5
            )

            if exit_short:
                return "CustomExitShort"

        # If no exit triggered:
        return None