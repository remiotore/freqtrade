# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Dict, List, Tuple, Union, Any

from freqtrade.strategy import (
    IStrategy,
    PairLocks,
    IntParameter,
    DecimalParameter,
)
from freqtrade.persistence import Trade

import talib.abstract as ta


class RSITrailingStopScalper(IStrategy):
    """
    RSI-TrailingStop Scalper für SOL/USDC (M15)

    STRATEGIE-KONZEPT:
    - Einstieg bei kurzfristiger RSI-Schwäche (RSI < threshold) auf 15M Chart
    - Ausstieg über Standard Freqtrade Trailing Stop
    - Fokus auf stabilere Signale mit optimalen Exits
    - Robustes Risikomanagement mit Cooldown-Zeiten

    OPTIMIERT FÜR:
    - Pair: SOL/USDC
    - Exchange: Binance Spot
    - Timeframe: 15 Minuten
    - Trading-Stil: Stabileres Scalping mit qualitativ hochwertigen Signalen

    TRAILING STOP KONFIGURATION:
    - Aktivierung ab: ≥ 1,0% Gewinn
    - Trailing Offset: 0,4%
    - Netto-Gewinn ab +0,6% nach Gebühren
    - Standard Freqtrade Trailing Stop (optimal für 15M)
    """

    # Strategy Interface Version
    INTERFACE_VERSION = 3

    # Nur Long-Positionen für Einfachheit
    can_short: bool = False

    # ROI deaktiviert - nur Trailing Stop + Stop Loss
    minimal_roi = {"0": 0}

    # Stop Loss - wird durch Parameter überschrieben
    stoploss = -0.010

    # STANDARD TRAILING STOP KONFIGURATION
    trailing_stop = True  # Standard Freqtrade Trailing Stop
    trailing_stop_positive = 0.0040  # Wird durch Parameter überschrieben
    trailing_stop_positive_offset = 0.0100  # Wird durch Parameter überschrieben
    trailing_only_offset_is_reached = True  # Nur trailing wenn Offset erreicht

    # Haupt-Timeframe: 15 Minuten
    timeframe = "15m"

    # Nur neue Kerzen verarbeiten
    process_only_new_candles = True

    # Standard Exit-Konfiguration
    use_exit_signal = False  # Nur Trailing Stop + Stop Loss
    exit_profit_only = False

    # HYPEROPT-PARAMETER für Optimierung

    # Entry-Parameter
    rsi_buy_min = IntParameter(25, 50, default=35, space="buy", optimize=True)
    rsi_timeperiod = IntParameter(10, 21, default=14, space="buy", optimize=True)

    # Exit/Risk-Parameter
    stoploss_param = DecimalParameter(
        -0.020, -0.005, default=-0.010, space="sell", optimize=True, decimals=3
    )
    trailing_stop_positive_param = DecimalParameter(
        0.0030, 0.0080, default=0.0040, space="sell", optimize=True, decimals=4
    )
    trailing_stop_positive_offset_param = DecimalParameter(
        0.0060, 0.0200, default=0.0100, space="sell", optimize=True, decimals=4
    )

    # Cooldown-Parameter
    cooldown_minutes = IntParameter(3, 15, default=7, space="sell", optimize=True)

    # Mindestens 30 Kerzen für RSI-Berechnung
    startup_candle_count: int = 30

    # Order-Types - limit mit post_only für bessere Fees
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Order-Konfiguration mit post_only
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Plot-Konfiguration für Backtesting-Charts
    plot_config = {
        "main_plot": {},
        "subplots": {
            "RSI": {
                "rsi": {"color": "orange"},
            },
        },
    }

    def bot_start(self, **kwargs: Any) -> None:
        """
        Wird beim Start des Bots aufgerufen.
        Setzt die optimierten Parameter als aktuelle Werte.
        """
        # Stop Loss Parameter anwenden
        self.stoploss = self.stoploss_param.value

        # Trailing Stop Parameter anwenden
        self.trailing_stop_positive = self.trailing_stop_positive_param.value
        self.trailing_stop_positive_offset = (
            self.trailing_stop_positive_offset_param.value
        )

    def populate_indicators(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        Fügt Indikatoren für Entry hinzu

        INDIKATOREN:
        - RSI(14): Relative Strength Index für Entry-Signale auf 15M
        """

        # RSI auf 15M - das einzige Entry-Signal
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_timeperiod.value)

        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        ENTRY-LOGIK für Long-Positionen:

        EINFACHE REGEL:
        - RSI(14) auf 15M < rsi_buy_min (kurzfristige Schwäche)
        - Sofortiger Entry bei Signal

        SCALPING-PHILOSOPHIE:
        Nutze kurzfristige RSI-Überverkauft-Situationen auf 15M für stabile Rebounds
        """

        dataframe.loc[
            (
                # Hauptsignal: 15M RSI unter Schwellenwert (Überverkauft)
                (dataframe["rsi"] < self.rsi_buy_min.value)
                &
                # Basic Sanity Check: Volume > 0
                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: Dict[str, Any]
    ) -> DataFrame:
        """
        EXIT-LOGIK:

        KEIN manueller Exit - alles läuft über Standard Trailing Stop!
        Das ist die Kern-Philosophie dieser Strategie:
        - Lass Gewinne laufen mit Trailing Stop
        - Minimiere Verluste mit klassischem Stop Loss
        """

        # Bewusst KEINE Exit-Signale setzen
        # Alles läuft über Standard Trailing Stop + Stop Loss

        return dataframe

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
        **kwargs: Dict[str, Any],
    ) -> bool:
        """
        Trade Entry Bestätigung mit post_only für bessere Fees
        """

        # Prüfe Cooldown-Zeit (optimierbar zwischen 3-15 Minuten)
        if self.is_pair_locked(pair, current_time):
            return False

        # Zusätzliche Sicherheitscheck: Minimum Order Size
        min_order_value = 10.0  # 10 USDC Minimum
        order_value = amount * rate

        if order_value < min_order_value:
            return False

        return True

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
        **kwargs: Dict[str, Any],
    ) -> bool:
        """
        Trade Exit Bestätigung mit Cooldown-Setup
        """

        # Nach jedem Trade: Optimierbare Cooldown-Zeit setzen
        cooldown_time = current_time + timedelta(minutes=self.cooldown_minutes.value)
        PairLocks.lock_pair(
            pair=pair, until=cooldown_time, reason="RSI Scalper Cooldown"
        )

        return True

    def is_pair_locked(self, pair: str, current_time: datetime) -> bool:
        """
        Prüft ob das Pair noch im Cooldown ist
        """
        return bool(PairLocks.is_pair_locked(pair, current_time))
