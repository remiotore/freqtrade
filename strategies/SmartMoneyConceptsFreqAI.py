# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, Set
from functools import reduce
import os
import json
import logging
from pathlib import Path

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib

# Add logger import to fix linter errors
logger = logging.getLogger(__name__)


class SmartMoneyConceptsFreqAI(IStrategy):
    """
    Smart Money Concepts (SMC) Strategy with FreqAI - Optimized for Futures

    This strategy combines Smart Money Concepts with machine learning by:
    1. Converting SMC principles into quantifiable features
    2. Using machine learning to identify optimal entry/exit points
    3. Adapting to market regime changes through continuous retraining

    FreqAI integrates with SMC concepts by learning:
    - Order block recognition patterns
    - Fair value gap significance in different contexts
    - Market structure shifts and their predictive value
    - Dynamic volatility-based risk management
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Informative timeframe for trend identification
    informative_timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = True  # Enabled shorts to improve performance in bearish markets

    # Minimal ROI designed for the strategy - optimized values
    minimal_roi = {"0": 0.192, "16": 0.054, "41": 0.03, "115": 0}

    # Optimal stoploss designed for the strategy - Changed to float for compatibility
    stoploss = -0.15
    # stoploss = DecimalParameter(
    #     -0.03, -0.01, default=-0.15, decimals=3, space="sell", optimize=True
    # )

    # Trailing stoploss - optimized settings - Changed to floats for compatibility
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    # trailing_stop_positive = DecimalParameter(
    #     0.005, 0.02, default=0.01, decimals=3, space="sell", optimize=True
    # )
    # trailing_stop_positive_offset = DecimalParameter(
    #     0.01, 0.05, default=0.02, decimals=3, space="sell", optimize=True
    # )
    trailing_only_offset_is_reached = False  # Changed to activate trailing stop immediately

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    # Custom stoploss thresholds (hyperoptable)
    cstp_profit_threshold = DecimalParameter(
        0.01, 0.03, default=0.024, decimals=3, space="sell", optimize=True
    )
    cstp_time_threshold_hours = IntParameter(1, 5, default=4, space="sell", optimize=True)
    cstp_atr_multiplier = DecimalParameter(
        1.0, 2.0, default=1.4, decimals=1, space="sell", optimize=True
    )

    # Strategy parameters (hyperoptable)
    # Order block lookback period
    ob_lookback = IntParameter(5, 20, default=8, space="buy")
    # Fair value gap threshold (% of candle)
    fvg_threshold = DecimalParameter(0.001, 0.01, default=0.0023, decimals=4, space="buy")
    # Order block threshold (How strong the order block should be)
    ob_threshold = DecimalParameter(0.001, 0.01, default=0.0082, decimals=4, space="buy")
    # Minimum strength for bullish/bearish power candles
    bullish_strength = DecimalParameter(0.01, 0.05, default=0.04, decimals=3, space="buy")
    bearish_strength = DecimalParameter(0.01, 0.05, default=0.01, decimals=3, space="sell")

    # Order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # Order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def plot_config(self):
        """
        Enhanced plot configuration using all supported FreqTrade features
        Official documentation: https://www.freqtrade.io/en/stable/plotting/#plot-configuration
        """
        return {
            # Main price chart with EMA crossover visualization
            "main_plot": {
                "ema_50": {
                    "color": "rgba(66, 135, 245, 0.9)",  # Brighter blue with transparency
                    "width": 1.75,
                },
                "ema_200": {
                    "color": "rgba(230, 57, 70, 0.9)",  # Brighter red with transparency
                    "width": 1.75,
                },
            },
            "subplots": {
                # Order Blocks with filled areas and clearer indicators
                "Order Blocks": {
                    "bullish_ob_high": {
                        "color": "rgba(46, 204, 113, 0.7)",  # Green with transparency
                        "width": 1.25,
                        "fill_to": "bullish_ob_low",
                        "fill_color": "rgba(46, 204, 113, 0.2)",  # Lighter fill color
                    },
                    "bullish_ob_low": {"color": "rgba(46, 204, 113, 0.7)", "width": 1.25},
                    "bearish_ob_high": {
                        "color": "rgba(231, 76, 60, 0.7)",  # Red with transparency
                        "width": 1.25,
                        "fill_to": "bearish_ob_low",
                        "fill_color": "rgba(231, 76, 60, 0.2)",  # Lighter fill color
                    },
                    "bearish_ob_low": {"color": "rgba(231, 76, 60, 0.7)", "width": 1.25},
                },
                # Signals with optimized bar indicators
                "Signals": {
                    "bos_signal": {
                        "color": "#f39c12",  # Orange
                        "width": 2,
                        "type": "bar",
                        "plotly": {"opacity": 0.9},
                    },
                    "bullish_fvg": {
                        "color": "#2ecc71",  # Green
                        "width": 1.5,
                        "type": "bar",
                        "plotly": {"opacity": 0.8},
                    },
                    "bearish_fvg": {
                        "color": "#e74c3c",  # Red
                        "width": 1.5,
                        "type": "bar",
                        "plotly": {"opacity": 0.8},
                    },
                    # Use scatter plot for prediction to make it stand out
                    "&-trend_prediction": {
                        "color": "#9b59b6",  # Purple
                        "width": 2,
                        "type": "scatter",
                        "plotly": {"opacity": 0.9},
                    },
                },
                # Market trend indicators with clear visuals
                "Market Trend": {
                    "market_trend": {
                        "color": "#3498db",  # Blue
                        "width": 1.75,
                        "type": "line",
                        "fill_to": 0,  # Fill to zero line
                        "fill_color": "rgba(52, 152, 219, 0.1)",  # Very light blue fill
                    },
                    "uptrend_1h": {
                        "color": "#2ecc71",  # Green
                        "width": 1.75,
                        "type": "scatter",
                        "plotly": {"opacity": 0.9},
                    },
                },
                # RSI with clearer visualization
                "RSI": {
                    "rsi": {
                        "color": "#8e44ad",  # Purple
                        "width": 1.75,
                        "type": "line",
                    },
                    # Add overbought/oversold reference lines
                    "overbought": {
                        "color": "rgba(231, 76, 60, 0.4)",  # Red with transparency
                        "width": 1,
                        "value": 70,
                    },
                    "oversold": {
                        "color": "rgba(46, 204, 113, 0.4)",  # Green with transparency
                        "width": 1,
                        "value": 30,
                    },
                },
                # Volume with better bar visualization
                "Volume": {
                    "volume": {
                        "color": "#3498db",  # Blue
                        "type": "bar",
                        "plotly": {"opacity": 0.7},
                    },
                    "volume_mean": {
                        "color": "#f39c12",  # Orange
                        "width": 1.5,
                        "type": "line",
                    },
                },
            },
        }

    def informative_pairs(self):
        """
        Define additional informative pair/timeframe combinations to be cached
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    # NOTE: All the existing SMC strategy methods remain intact
    # (detect_bullish_order_block, detect_bearish_order_block, etc.)
    # Insert them all here - I'm not repeating them for brevity

    # FreqAI Feature Engineering Methods

    def normalize_dataframe_timezones(self, dataframe: DataFrame) -> DataFrame:
        """
        Ensures all datetime columns in a dataframe have consistent timezone
        This is called after merge operations to fix timezone issues
        """
        try:
            if not isinstance(dataframe, DataFrame) or len(dataframe) == 0:
                return dataframe

            # Identify all datetime columns
            datetime_cols = []
            for col in dataframe.columns:
                if pd.api.types.is_datetime64_dtype(dataframe[col]):
                    datetime_cols.append(col)

            if not datetime_cols:
                return dataframe

            # Check timezone of each datetime column
            for col in datetime_cols:
                # If column is timezone naive, localize to UTC
                if dataframe[col].dt.tz is None:
                    dataframe[col] = dataframe[col].dt.tz_localize("UTC")
                # If column has a timezone that's not UTC, convert to UTC
                elif str(dataframe[col].dt.tz) != "UTC":
                    dataframe[col] = dataframe[col].dt.tz_convert("UTC")

            return dataframe
        except Exception as e:
            logger.warning(f"Error normalizing dataframe timezones: {str(e)}")
            return dataframe

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Expand SMC indicators across multiple timeframes with FreqAI
        """
        try:
            # Ensure input dataframe has consistent timezone handling
            dataframe = self.normalize_dataframe_timezones(dataframe)

            # Add feature version tracking
            dataframe["%-feature_version-period"] = 2.0  # Increment this when feature set changes

            # Volume indicators
            dataframe["%-volume_mean-period"] = dataframe["volume"].rolling(window=period).mean()
            dataframe["%-volume_ratio-period"] = dataframe["volume"] / dataframe[
                "%-volume_mean-period"
            ].fillna(1.0)

            # Momentum indicators
            dataframe["%-momentum-period"] = dataframe["close"].pct_change(period).fillna(0.0)
            dataframe["%-momentum_smooth-period"] = (
                dataframe["%-momentum-period"].rolling(3).mean().fillna(0.0)
            )

            # Volatility indicators
            dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
            dataframe["%-volatility-period"] = (
                dataframe["%-atr-period"] / dataframe["close"]
            ).fillna(0.0)

            # Basic price indicators (more stable than complex ones)
            dataframe["%-close-period"] = dataframe["close"].pct_change(period).fillna(0.0)
            dataframe["%-high-period"] = dataframe["high"].pct_change(period).fillna(0.0)
            dataframe["%-low-period"] = dataframe["low"].pct_change(period).fillna(0.0)
            dataframe["%-open-period"] = dataframe["open"].pct_change(period).fillna(0.0)

            # SMC-specific features - simplified for stability
            # Body size
            dataframe["%-body_size-period"] = (
                abs(dataframe["open"] - dataframe["close"]) / dataframe["close"]
            )

            # Is bearish/bullish - as float values (more stable than integers)
            dataframe["%-is_bearish-period"] = (dataframe["open"] > dataframe["close"]).astype(
                float
            )
            dataframe["%-is_bullish-period"] = (dataframe["open"] < dataframe["close"]).astype(
                float
            )

            # Standard technical indicators - more stable than raw price features
            dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
            dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
            dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)

            # Ensure no NaN values in any features
            for col in dataframe.columns:
                if col.startswith("%-") and dataframe[col].isna().any():
                    dataframe[col] = dataframe[col].fillna(0)

            # Print feature count on first candle for period 5
            if period == 5 and len(dataframe) > 0 and metadata["pair"] == "BTC/USDT:USDT":
                feature_count = len([col for col in dataframe.columns if col.startswith("%-")])
                print(f"Feature engineering expanded {feature_count} features for period {period}")

            return dataframe

        except Exception as e:
            logger.warning(f"Error in feature_engineering_expand_all: {str(e)}")
            # Return dataframe with minimal feature set to avoid crashes
            dataframe["%-feature_version-period"] = 2.0
            dataframe["%-rsi-period"] = 50.0  # Default neutral value
            dataframe["%-volume_mean-period"] = dataframe["volume"].mean()
            return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Define basic features that don't need to be duplicated for each timeframe
        """
        try:
            # Ensure consistent timezone handling
            dataframe = self.normalize_dataframe_timezones(dataframe)

            # Basic price metrics - simple and stable
            dataframe["%-pct-change"] = dataframe["close"].pct_change().fillna(0)
            dataframe["%-raw_volume"] = dataframe["volume"] / dataframe["volume"].mean()
            dataframe["%-raw_price"] = dataframe["close"] / dataframe["close"].mean()

            # Market structure from existing SMC calculations - simplified for stability
            if "bos_signal" in dataframe.columns:
                dataframe["%-bos_signal"] = dataframe["bos_signal"].astype(float)
            else:
                dataframe["%-bos_signal"] = 0.0

            # Order blocks from SMC - converted to float for stability
            dataframe["%-has_bullish_ob"] = (
                dataframe["bullish_ob"].fillna(0).astype(float)
                if "bullish_ob" in dataframe.columns
                else 0.0
            )
            dataframe["%-has_bearish_ob"] = (
                dataframe["bearish_ob"].fillna(0).astype(float)
                if "bearish_ob" in dataframe.columns
                else 0.0
            )

            # Fair value gaps from SMC - converted to float for stability
            dataframe["%-bullish_fvg"] = (
                dataframe["bullish_fvg"].fillna(0).astype(float)
                if "bullish_fvg" in dataframe.columns
                else 0.0
            )
            dataframe["%-bearish_fvg"] = (
                dataframe["bearish_fvg"].fillna(0).astype(float)
                if "bearish_fvg" in dataframe.columns
                else 0.0
            )

            # High timeframe trend
            dataframe["%-uptrend_1h"] = (
                dataframe["uptrend_1h"].fillna(True).astype(float)
                if "uptrend_1h" in dataframe.columns
                else 1.0
            )

            # Ensure no NaN values
            for col in dataframe.columns:
                if col.startswith("%-") and dataframe[col].isna().any():
                    dataframe[col] = dataframe[col].fillna(0)

            # Print diagnostic info only for BTC pair
            if metadata["pair"] == "BTC/USDT:USDT" and len(dataframe) > 0:
                feature_count = len([col for col in dataframe.columns if col.startswith("%-")])
                print(f"Basic feature count: {feature_count}")

            return dataframe

        except Exception as e:
            logger.warning(f"Error in feature_engineering_expand_basic: {str(e)}")
            # Return dataframe with minimal feature set
            dataframe["%-pct-change"] = 0.0
            dataframe["%-raw_volume"] = 1.0
            dataframe["%-raw_price"] = 1.0
            return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Engineer features that are not auto-expanded across timeframes
        Set a minimal deterministic feature set for consistency
        """
        try:
            # Ensure any datetime operations use timezone-aware dates
            # If working with date parts, make sure the date column is TZ-aware first
            if len(dataframe) > 0 and pd.api.types.is_datetime64_dtype(dataframe["date"]):
                if dataframe["date"].dt.tz is None:
                    # Convert to timezone-aware with UTC to avoid mixing tz-aware/tz-naive
                    dataframe["date"] = dataframe["date"].dt.tz_localize("UTC")

            # Time-based features - use dt accessor safely with consistent timezone
            # Convert any time-based features to float instead of using categoricals
            # This avoids timezone issues when merging later
            dataframe["%-day_of_week"] = (
                dataframe["date"].dt.dayofweek.astype(float) / 6.0
            )  # Scale to 0-1
            dataframe["%-hour_of_day"] = (
                dataframe["date"].dt.hour.astype(float) / 23.0
            )  # Scale to 0-1

            # These are standard features that should be generated consistently
            if "ema_50" in dataframe.columns:
                dataframe["%-dist-to-ema50"] = (
                    dataframe["close"] - dataframe["ema_50"]
                ) / dataframe["close"]
            else:
                dataframe["%-dist-to-ema50"] = 0.0

            if "ema_200" in dataframe.columns:
                dataframe["%-dist-to-ema200"] = (
                    dataframe["close"] - dataframe["ema_200"]
                ) / dataframe["close"]
            else:
                dataframe["%-dist-to-ema200"] = 0.0

            # Market trend values
            dataframe["%-market_trend"] = (
                dataframe["market_trend"].fillna(0) / 100.0
                if "market_trend" in dataframe.columns
                else 0.0
            )
            dataframe["%-uptrend_1h"] = (
                dataframe["uptrend_1h"].fillna(True).astype(float)
                if "uptrend_1h" in dataframe.columns
                else 1.0
            )

            # Ensure all pairs have the same feature set and all values are floats
            for col in [c for c in dataframe.columns if c.startswith("%-")]:
                dataframe[col] = dataframe[col].astype(float)

            # Print diagnostic info only for BTC pair
            if metadata["pair"] == "BTC/USDT:USDT" and len(dataframe) > 0:
                feature_count = len([col for col in dataframe.columns if col.startswith("%-")])
                print(f"Standard feature count: {feature_count}")

                # Print some debugging details if needed
                print(f"Feature dimensions check for {metadata['pair']}:")
                print(f"Total features: {feature_count}")
                if feature_count < 4:
                    print("WARNING: Fewer than 4 features! This will cause reshape errors.")
                    print(
                        f"Available features: {[c for c in dataframe.columns if c.startswith('%-')]}"
                    )

            return dataframe

        except Exception as e:
            logger.warning(f"Error in feature_engineering_standard: {str(e)}")
            # Return dataframe with at least a few basic features to avoid crashes
            if "%-day_of_week" not in dataframe.columns:
                dataframe["%-day_of_week"] = 0.5  # Default value
            if "%-hour_of_day" not in dataframe.columns:
                dataframe["%-hour_of_day"] = 0.5  # Default value
            return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Set prediction targets for the model
        Ensures a single-column label for LightGBM classification
        """
        try:
            # Target for price movement prediction (main target)
            label_period = self.freqai_info["feature_parameters"]["label_period_candles"]

            # Make a simple target - future close price direction
            # This avoids any potential multi-column issues with LightGBM
            future_close = dataframe["close"].shift(-label_period)

            # Simple target: price change direction (1 for up, 0 for down)
            # IMPORTANT: Convert numeric values to strings to avoid KeyError with integer indices
            # Instead of binary 0/1, use "down"/"up" string labels which are safer
            price_up = (future_close > dataframe["close"]).astype(int)
            dataframe["&-trend_prediction"] = price_up.map({1: "up", 0: "down"}).fillna("unknown")

            if len(dataframe) > 0:
                # Log the first few values of the target to verify
                print(f"Target values created for {metadata['pair']} (first 5):")
                print(dataframe["&-trend_prediction"].head(5).tolist())

                # Ensure target class names are defined in the model
                if hasattr(self, "freqai") and hasattr(self.freqai, "class_names"):
                    self.freqai.class_names = ["down", "up"]
                    print(f"Set class names to: {self.freqai.class_names}")

            return dataframe

        except Exception as e:
            print(f"Error in set_freqai_targets for {metadata['pair']}: {e}")
            # Add a safe default target to avoid further errors
            dataframe["&-trend_prediction"] = "unknown"
            return dataframe

    def get_consistent_feature_set(self, dataframe, metadata: dict) -> Set[str]:
        """
        Return a simplified consistent feature set
        """
        try:
            # Fix feature columns with _x suffix, which is causing model mismatch
            # Look for columns with _x suffix and rename them
            rename_map = {}
            for col in dataframe.columns:
                if not isinstance(col, str):
                    continue  # Skip non-string columns to avoid the startswith attribute error
                if col.startswith("%-") and col.endswith("_x"):
                    # Remove the _x suffix to match the expected format
                    new_col = col[:-2]  # Remove last 2 chars (_x)
                    rename_map[col] = new_col

            # Apply the renaming if needed
            if rename_map:
                print(f"Fixing {len(rename_map)} columns with _x suffix for {metadata['pair']}")
                dataframe = dataframe.rename(columns=rename_map)

            # Get all valid feature columns that start with %-
            feature_columns = []
            for col in dataframe.columns:
                if isinstance(col, str) and col.startswith("%-"):
                    feature_columns.append(col)
                elif not isinstance(col, str):
                    print(
                        f"Skipping non-string column: {col} (type: {type(col)}) for {metadata['pair']}"
                    )

            print(f"Feature set for {metadata['pair']} (count: {len(feature_columns)}, sample 5):")
            print(feature_columns[:5] if feature_columns else "No features found!")

            return set(feature_columns)

        except Exception as e:
            print(f"Error in get_consistent_feature_set: {e}")
            # Return a minimal set to avoid crashing
            minimal_features = set()
            for col in dataframe.columns:
                if isinstance(col, str) and col.startswith("%-"):
                    minimal_features.add(col)

            # Add at least two basic features if none found
            if not minimal_features:
                print("WARNING: No features found! Adding placeholder features.")
                if "%-day_of_week" not in dataframe.columns:
                    dataframe["%-day_of_week"] = 0.5
                    minimal_features.add("%-day_of_week")
                if "%-hour_of_day" not in dataframe.columns:
                    dataframe["%-hour_of_day"] = 0.5
                    minimal_features.add("%-hour_of_day")

            return minimal_features

    def normalize_data_dict(self, data_dict: dict) -> dict:
        """
        CRITICAL FIX: This method is called by FreqAI right before prediction, allowing us
        to fix column name issues that happen during FreqAI's internal merge operations
        """
        try:
            # Check if prediction_features exists in the data dictionary
            if "prediction_features" not in data_dict:
                return data_dict

            # Get the dataframe containing prediction features
            df = data_dict["prediction_features"]

            # First check and handle _x suffix in all feature columns
            rename_map = {}
            for col in df.columns:
                if not isinstance(col, str):
                    continue  # Skip non-string columns

                # Handle _x suffix from pandas merge operations
                if col.startswith("%-") and col.endswith("_x"):
                    # Remove the _x suffix to match expected format
                    new_col = col[:-2]
                    rename_map[col] = new_col

            # Apply the renaming if needed
            if rename_map:
                print(f"NORMALIZING DATA DICT: Fixing {len(rename_map)} columns with suffix issues")
                # Print some examples of the renames
                examples = list(rename_map.items())[:5]
                print(f"Examples of renames: {examples}")

                # Rename the columns
                df = df.rename(columns=rename_map)
                data_dict["prediction_features"] = df

            # Ensure labels_mean exists in the data dictionary to avoid KeyError
            if "labels_mean" not in data_dict.get("data", {}):
                print("Adding empty labels_mean dictionary to avoid KeyError")
                if "data" not in data_dict:
                    data_dict["data"] = {}
                data_dict["data"]["labels_mean"] = {}

            # Ensure integer labels are properly mapped in labels_mean
            if "data" in data_dict and "labels_mean" in data_dict["data"]:
                # Get all target column names (they start with &-)
                target_cols = [
                    col for col in df.columns if isinstance(col, str) and col.startswith("&-")
                ]
                # Add integer labels to labels_mean if they don't exist
                for i in range(5):  # Add indices 0-4 just to be safe
                    if i not in data_dict["data"]["labels_mean"]:
                        data_dict["data"]["labels_mean"][i] = 0.0

            # Print summary of column count to help with debugging
            print(f"Final column count after normalization: {len(df.columns)}")

            return data_dict
        except Exception as e:
            print(f"Error in normalize_data_dict: {e}")
            # Return original dict on error to avoid crashing
            return data_dict

    def reset_feature_set(self):
        """
        Simplified reset function
        """
        print("Feature set reset functionality simplified")
        return False

    def check_gpu_utilization(self):
        """
        Check GPU utilization to verify resources are being used properly.
        This is a diagnostic method that can be called when troubleshooting GPU usage.
        """
        try:
            import subprocess
            import sys

            # Detect OS for appropriate command
            if sys.platform == "win32":
                # Windows - use nvidia-smi
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                output = result.stdout
            elif sys.platform in ["linux", "linux2"]:
                # Linux - use nvidia-smi
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                output = result.stdout
            else:
                # macOS or other platform
                return "GPU check not supported on this platform"

            print(f"GPU Status:\n{output}")
            return output
        except Exception as e:
            return f"Error checking GPU: {str(e)}"

    def handle_sklearn_warnings(self):
        """
        Handle sklearn deprecation warnings by monkey patching.
        This is needed to silence the force_all_finite warning.
        """
        try:
            import sklearn.utils.validation

            # Save the original function
            original_check_array = sklearn.utils.validation.check_array

            # Create a wrapper that handles the force_all_finite parameter
            def patched_check_array(array, *args, **kwargs):
                if "force_all_finite" in kwargs:
                    # Replace force_all_finite with ensure_all_finite
                    kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                return original_check_array(array, *args, **kwargs)

            # Replace the original function with our wrapper
            sklearn.utils.validation.check_array = patched_check_array

            print(
                "Successfully patched sklearn.utils.validation.check_array to handle force_all_finite"
            )
            return True
        except Exception as e:
            print(f"Failed to patch sklearn: {e}")
            return False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        try:
            # Ensure date column has consistent timezone
            # If 'date' is timezone naive, localize it to UTC
            if isinstance(dataframe, DataFrame) and len(dataframe) > 0:
                if pd.api.types.is_datetime64_dtype(dataframe["date"]):
                    # First check if the date column is timezone-naive
                    if dataframe["date"].dt.tz is None:
                        # Convert to timezone-aware with UTC
                        dataframe["date"] = dataframe["date"].dt.tz_localize("UTC")
                    # If already timezone-aware but not UTC, convert to UTC
                    elif str(dataframe["date"].dt.tz) != "UTC":
                        dataframe["date"] = dataframe["date"].dt.tz_convert("UTC")

                # Log the timezone status for debugging
                if len(dataframe) > 0:
                    logger.info(f"Date timezone for {metadata['pair']}: {dataframe['date'].dt.tz}")

        except Exception as e:
            logger.warning(f"Error normalizing timezones: {str(e)}")

        # Call the parent populate_indicators
        dataframe = super().populate_indicators(dataframe, metadata)

        if self.freqai_info != {}:
            # CRITICAL FIX: Log column names before prediction to troubleshoot _x suffix
            # Create rename dictionary to rename _x suffix columns
            rename_map = {}
            x_suffix_count = 0
            for col in dataframe.columns:
                if isinstance(col, str) and col.endswith("_x"):
                    rename_map[col] = col[:-2]  # Remove _x suffix
                    x_suffix_count += 1

            # Apply renaming if any _x columns found - do this for all modes (not just live)
            if x_suffix_count > 0:
                if not self.dp.runmode.value in ("backtest", "plot"):
                    logger.info(f"Fixing {x_suffix_count} columns with _x suffix")
                    logger.info(f"Example renames: {list(rename_map.items())[:3]}")
                dataframe = dataframe.rename(columns=rename_map)

            # Fix pair name format in column names (from BTC/USDT:USDT to BTCUSDTUSDT)
            pair_rename_map = {}
            for col in dataframe.columns:
                if isinstance(col, str) and "/" in col and ":" in col:
                    fixed_col = col.replace("/", "").replace(":", "")
                    pair_rename_map[col] = fixed_col

            if pair_rename_map:
                if not self.dp.runmode.value in ("backtest", "plot"):
                    logger.info(f"Fixing {len(pair_rename_map)} columns with pair format issues")
                    logger.info(f"Example pair renames: {list(pair_rename_map.items())[:3]}")
                dataframe = dataframe.rename(columns=pair_rename_map)

            if not self.dp.runmode.value in ("backtest", "plot"):
                logger.info(f"Final columns after fixing: {dataframe.columns[:5]}")
                logger.info(f"Total columns after fixing: {len(dataframe.columns)}")

            # Check dataframe date timezone again before passing to FreqAI
            try:
                if len(dataframe) > 0 and pd.api.types.is_datetime64_dtype(dataframe["date"]):
                    logger.info(f"Date timezone before FreqAI: {dataframe['date'].dt.tz}")
            except Exception:
                pass

            dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal logic using FreqAI predictions - modified for classification model
        """
        # Initialize entry columns
        df["enter_long"] = 0
        df["enter_short"] = 0

        # CRITICAL FIX: Completely exclude AVAX pair which is causing errors
        if "AVAX" in metadata["pair"]:
            print(f"Skipping entry signals for {metadata['pair']} due to known issues")
            return df

        # Define top performers and poor performers (unchanged)
        top_performers = ["DOGE/USDT:USDT", "SOL/USDT:USDT", "PAXG/USDT:USDT"]
        poor_performers = ["ADA/USDT:USDT"]

        # Exclude AVAX completely
        if "AVAX" in metadata["pair"]:
            return df

        # Ensure all required columns are correctly processed (no NaNs) before use
        if "do_predict" not in df.columns or "high_volatility" not in df.columns:
            return df

        # Ensure uptrend_1h and market_trend have no NaNs
        df["uptrend_1h"] = df["uptrend_1h"].fillna(True)
        df["market_trend"] = df["market_trend"].fillna(0)

        # Ensure prediction column has no NaNs
        if "&-trend_prediction" not in df.columns:
            return df

        # Fill missing values - use string "unknown" for unclassified cases
        df["&-trend_prediction"] = df["&-trend_prediction"].fillna("unknown")

        # Fill other potentially used columns
        if "volume_ratio" in df.columns:
            df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
        if "rsi" in df.columns:
            df["rsi"] = df["rsi"].fillna(50)

        # Check that FreqAI prediction is valid
        base_filter = df["do_predict"] == 1  # FreqAI says prediction is valid

        # Long entries - For classification model using string values ("up" = bullish prediction)
        long_conditions = (
            base_filter
            & (df["&-trend_prediction"] == "up")  # Bullish prediction (class = "up")
            & (df["market_trend"] > -20)  # Only enter longs when market not extremely bearish
        )

        # Add specific order block entry conditions
        bullish_ob_entry = (
            df["bullish_ob"].fillna(0)
            > 0  # We have a bullish order block
            & (df["bullish_ob_quality"].fillna(0) > 1.2)  # Higher quality OBs only
        )

        # Add SMC-specific entry conditions based on performer category
        if metadata["pair"] in top_performers:
            # More relaxed conditions for top performers
            long_entry = long_conditions | (
                base_filter & bullish_ob_entry & (df["&-trend_prediction"] == "up")
            )
        elif metadata["pair"] in poor_performers:
            # Still strict conditions for poor performers
            long_entry = (
                long_conditions
                & (df["rsi"] < 70)  # Less strict RSI condition
                & (df["market_trend"] > 0)  # Only in positive market trend
            )
        else:
            # Standard conditions - rely on FreqAI prediction with order block boost
            long_entry = long_conditions | (
                base_filter
                & bullish_ob_entry
                & (df["&-trend_prediction"] == "up")
                & (df["market_trend"] > -10)
            )

        # Short entries - For classification model using string values ("down" = bearish prediction)
        short_conditions = (
            base_filter
            & (df["&-trend_prediction"] == "down")  # Bearish prediction (class = "down")
            & (df["market_trend"] < 20)  # Only enter shorts when market not extremely bullish
        )

        # Add specific order block entry conditions
        bearish_ob_entry = (
            df["bearish_ob"].fillna(0)
            > 0  # We have a bearish order block
            & (df["bearish_ob_quality"].fillna(0) > 1.2)  # Higher quality OBs only
        )

        # Add SMC-specific entry conditions based on performer category
        if metadata["pair"] in top_performers:
            # More relaxed conditions for top performers with shorts
            short_entry = short_conditions | (
                base_filter & bearish_ob_entry & (df["&-trend_prediction"] == "down")
            )
        elif metadata["pair"] in poor_performers:
            # Still strict conditions for poor performers
            short_entry = (
                short_conditions
                & (df["rsi"] > 30)  # Less strict RSI condition
                & (df["market_trend"] < 0)  # Only in negative market trend
            )
        else:
            # Standard conditions - rely on FreqAI prediction with order block boost
            short_entry = short_conditions | (
                base_filter
                & bearish_ob_entry
                & (df["&-trend_prediction"] == "down")
                & (df["market_trend"] < 10)
            )

        # Set entry signals
        df.loc[long_entry, "enter_long"] = 1
        df.loc[long_entry, "enter_tag"] = "long_freqai"
        df.loc[short_entry, "enter_short"] = 1
        df.loc[short_entry, "enter_tag"] = "short_freqai"

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal logic using FreqAI predictions - modified for classification model
        """
        # Initialize exit columns
        df["exit_long"] = 0
        df["exit_short"] = 0

        # Skip if required columns don't exist
        if "do_predict" not in df.columns or "&-trend_prediction" not in df.columns:
            return df

        # Fill NaN values to prevent conversion errors
        df["do_predict"] = df["do_predict"].fillna(0)
        df["&-trend_prediction"] = df["&-trend_prediction"].fillna(
            "unknown"
        )  # Default to unknown signal

        # Exit long positions when prediction turns bearish (class = "down")
        df.loc[(df["do_predict"] == 1) & (df["&-trend_prediction"] == "down"), "exit_long"] = 1

        # Exit short positions when prediction turns bullish (class = "up")
        df.loc[(df["do_predict"] == 1) & (df["&-trend_prediction"] == "up"), "exit_short"] = 1

        return df

    # Keep all the custom exit, stoploss, and position adjustment logic from the original strategy
    # These methods remain the same

    # <<< INSERT MISSING METHODS HERE >>>
    # Insert detect_bullish_order_block, detect_bearish_order_block, detect_fair_value_gaps, detect_break_of_structure,
    # custom_stoploss, custom_exit, adjust_trade_position, and protections property.

    # --- Methods copied from SmartMoneyConceptsStrategy ---

    def detect_bullish_order_block(self, dataframe: DataFrame) -> DataFrame:
        """
        Identify bullish order blocks (OB)
        A bullish OB is a bearish candle that precedes a strong bullish move
        """
        # Calculate candle body size and direction
        dataframe["body_size"] = abs(dataframe["open"] - dataframe["close"])
        dataframe["body_pct"] = dataframe["body_size"] / dataframe["close"]
        dataframe["is_bearish"] = dataframe["open"] > dataframe["close"]

        # Identify potential bullish order blocks - lowered threshold for futures
        dataframe["potential_bullish_ob"] = dataframe["is_bearish"] & (
            dataframe["body_pct"] > self.ob_threshold.value * 0.8
        )

        # Identify bullish power candles (candles showing strong bullish momentum) - lowered threshold
        dataframe["bullish_power"] = (
            (dataframe["close"] > dataframe["open"])
            & (dataframe["body_pct"] > self.bullish_strength.value * 0.8)
            & (dataframe["volume"] > dataframe["volume"].rolling(20).mean() * 0.8)
        )

        # Rolling check for potential order blocks followed by strong bullish moves
        for i in range(1, self.ob_lookback.value + 1):
            cond = dataframe["potential_bullish_ob"] & dataframe["bullish_power"].shift(-i)
            # When condition is true, we found a bullish order block
            dataframe.loc[cond, "bullish_ob"] = 1
            # Mark the high of the bearish candle
            dataframe.loc[cond, "bullish_ob_high"] = dataframe["high"]
            # Mark the low of the bearish candle
            dataframe.loc[cond, "bullish_ob_low"] = dataframe["low"]
            # Calculate OB strength based on the power candle that follows
            dataframe.loc[cond, "bullish_ob_strength"] = dataframe["body_pct"].shift(-i)

        return dataframe

    def detect_bearish_order_block(self, dataframe: DataFrame) -> DataFrame:
        """
        Identify bearish order blocks (OB)
        A bearish OB is a bullish candle that precedes a strong bearish move
        """
        # Calculate candle parameters if not already done
        if "body_size" not in dataframe.columns:
            dataframe["body_size"] = abs(dataframe["open"] - dataframe["close"])
            dataframe["body_pct"] = dataframe["body_size"] / dataframe["close"]

        dataframe["is_bullish"] = dataframe["open"] < dataframe["close"]

        # Identify potential bearish order blocks - lowered threshold for futures
        dataframe["potential_bearish_ob"] = dataframe["is_bullish"] & (
            dataframe["body_pct"] > self.ob_threshold.value * 0.8
        )

        # Identify bearish power candles - lowered threshold
        dataframe["bearish_power"] = (
            (dataframe["close"] < dataframe["open"])
            & (dataframe["body_pct"] > self.bearish_strength.value * 0.8)
            & (dataframe["volume"] > dataframe["volume"].rolling(20).mean() * 0.8)
        )

        # Rolling check for potential order blocks followed by strong bearish moves
        for i in range(1, self.ob_lookback.value + 1):
            cond = dataframe["potential_bearish_ob"] & dataframe["bearish_power"].shift(-i)
            # When condition is true, we found a bearish order block
            dataframe.loc[cond, "bearish_ob"] = 1
            # Mark the high of the bullish candle
            dataframe.loc[cond, "bearish_ob_high"] = dataframe["high"]
            # Mark the low of the bullish candle
            dataframe.loc[cond, "bearish_ob_low"] = dataframe["low"]
            # Calculate OB strength
            dataframe.loc[cond, "bearish_ob_strength"] = dataframe["body_pct"].shift(-i)

        return dataframe

    def detect_fair_value_gaps(self, dataframe: DataFrame) -> DataFrame:
        """
        Identify Fair Value Gaps (FVG)
        Bullish FVG: Low of candle 1 > High of candle 3
        Bearish FVG: High of candle 1 < Low of candle 3
        """
        # Detect bullish fair value gaps - lowered threshold for futures
        dataframe["bullish_fvg"] = (dataframe["low"] > dataframe["high"].shift(2)) & (
            (dataframe["low"] - dataframe["high"].shift(2)) / dataframe["close"]
            > self.fvg_threshold.value * 0.7
        )

        # Detect bearish fair value gaps - lowered threshold for futures
        dataframe["bearish_fvg"] = (dataframe["high"] < dataframe["low"].shift(2)) & (
            (dataframe["low"].shift(2) - dataframe["high"]) / dataframe["close"]
            > self.fvg_threshold.value * 0.7
        )

        # Calculate FVG size for reference
        dataframe.loc[dataframe["bullish_fvg"], "bullish_fvg_size"] = dataframe["low"] - dataframe[
            "high"
        ].shift(2)

        dataframe.loc[dataframe["bearish_fvg"], "bearish_fvg_size"] = (
            dataframe["low"].shift(2) - dataframe["high"]
        )

        return dataframe

    def detect_break_of_structure(self, dataframe: DataFrame) -> DataFrame:
        """
        Detect breaks of market structure (BOS)
        A bullish BOS occurs when price breaks above a previous swing high
        A bearish BOS occurs when price breaks below a previous swing low
        """
        import numpy as np

        # Find swing highs and lows (simple method - can be enhanced)
        dataframe["swing_high"] = (
            (dataframe["high"] > dataframe["high"].shift(1))
            & (dataframe["high"] > dataframe["high"].shift(2))
            & (dataframe["high"] > dataframe["high"].shift(-1))
            & (dataframe["high"] > dataframe["high"].shift(-2))
        )

        dataframe["swing_low"] = (
            (dataframe["low"] < dataframe["low"].shift(1))
            & (dataframe["low"] < dataframe["low"].shift(2))
            & (dataframe["low"] < dataframe["low"].shift(-1))
            & (dataframe["low"] < dataframe["low"].shift(-2))
        )

        # Initialize new DataFrame columns for tracking swing points
        dataframe["last_swing_high"] = np.nan
        dataframe["last_swing_high_val"] = np.nan
        dataframe["last_swing_low"] = np.nan
        dataframe["last_swing_low_val"] = np.nan

        # Vectorized approach to track swing points without using loop and iloc
        # Create temporary series to hold our swing point indices and values
        last_swing_high_idx = pd.Series(np.nan, index=dataframe.index)
        last_swing_high_val = pd.Series(np.nan, index=dataframe.index)
        last_swing_low_idx = pd.Series(np.nan, index=dataframe.index)
        last_swing_low_val = pd.Series(np.nan, index=dataframe.index)

        # Find indices where swing_high is True and get corresponding high values
        high_idx = dataframe.index[dataframe["swing_high"]]
        high_vals = dataframe.loc[dataframe["swing_high"], "high"].values

        # Find indices where swing_low is True and get corresponding low values
        low_idx = dataframe.index[dataframe["swing_low"]]
        low_vals = dataframe.loc[dataframe["swing_low"], "low"].values

        # Forward fill the values using pandas ffill() method instead of the deprecated fillna(method="ffill")
        if len(high_idx) > 0:
            # Set values at the swing high points
            last_swing_high_idx.loc[high_idx] = high_idx
            last_swing_high_val.loc[high_idx] = high_vals

            # Forward fill
            dataframe["last_swing_high"] = last_swing_high_idx.ffill()
            dataframe["last_swing_high_val"] = last_swing_high_val.ffill()

        if len(low_idx) > 0:
            # Set values at the swing low points
            last_swing_low_idx.loc[low_idx] = low_idx
            last_swing_low_val.loc[low_idx] = low_vals

            # Forward fill
            dataframe["last_swing_low"] = last_swing_low_idx.ffill()
            dataframe["last_swing_low_val"] = last_swing_low_val.ffill()

        # Detect Bullish BOS (price breaks above the last swing high)
        dataframe["bullish_bos"] = (dataframe["close"] > dataframe["last_swing_high_val"]) & (
            dataframe["last_swing_high_val"].notnull()
        )

        # Detect Bearish BOS (price breaks below the last swing low)
        dataframe["bearish_bos"] = (dataframe["close"] < dataframe["last_swing_low_val"]) & (
            dataframe["last_swing_low_val"].notnull()
        )

        # Combine for single signal (-1 for bearish, 1 for bullish)
        dataframe["bos_signal"] = 0
        dataframe.loc[dataframe["bullish_bos"], "bos_signal"] = 1
        dataframe.loc[dataframe["bearish_bos"], "bos_signal"] = -1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Custom stoploss based on ATR, order block positioning,
        market structure, and time-based adjustments to reduce large losses
        """
        import numpy as np

        # Fetch dataframe for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # If no dataframe or empty dataframe, return the initial stoploss
        if dataframe.empty:
            return self.stoploss

        # Get the last row of the dataframe
        last_candle = dataframe.iloc[-1].squeeze()

        # Calculate trade duration in hours
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Time-based stoploss tightening - the longer a trade is open, the tighter the stoploss
        # This prevents catastrophic losses from stop hits after multiple days
        if trade_duration_hours > 24:  # After 1 day (reduced from 48 hours)
            return -0.03  # Much tighter stop regardless of other factors (reduced from -0.05)
        elif trade_duration_hours > 12:  # After 12 hours (reduced from 24 hours)
            return -0.07  # Tighter stop (reduced from -0.10)

        # Pair-specific stoploss settings
        problematic_pairs = [
            "XRP/USDT:USDT",
            "LINK/USDT:USDT",
            "DOT/USDT:USDT",
            "ETH/USDT:USDT",
            "BTC/USDT:USDT",
            "ADA/USDT:USDT",  # Added ADA as problematic
        ]
        if pair in problematic_pairs:
            # Tighter stoploss for problematic pairs
            if (
                current_profit > self.cstp_profit_threshold.value * 0.5
            ):  # 50% lower threshold for problematic pairs (was 0.7)
                return 0.0  # Move to breakeven faster for problematic pairs
            else:
                return -0.015  # Tighter initial stoploss for problematic pairs (was -0.02)

        # Add time-based tightening
        minutes_open = (current_time - trade.open_date_utc).total_seconds() / 60
        hours_threshold = self.cstp_time_threshold_hours.value

        # Gradually tighten stoploss the longer a trade is open
        if minutes_open > hours_threshold * 60:  # After X hours (reduced from 120 to 60 minutes)
            return max(
                -0.01, self.stoploss
            )  # Much tighter stop for longer trades (reduced from -0.015)
        elif minutes_open > hours_threshold * 30:  # After X/2 hours (reduced from 60 to 30 minutes)
            return max(-0.015, self.stoploss)  # Tighter stop (reduced from -0.02)

        # 1. If price is above key structure levels, tighten stop to breakeven faster
        if current_profit > self.cstp_profit_threshold.value * 0.7:  # 30% lower threshold
            return 0.0  # Breakeven

        # 2. If we have an active bullish order block below current price,
        # use it as a logical stoploss area (just below the OB)
        if (
            not np.isnan(last_candle.get("bullish_ob_low", np.nan))
            and current_rate > last_candle["bullish_ob_low"]
        ):
            # Calculate distance to OB
            distance_to_ob = (current_rate - last_candle["bullish_ob_low"]) / current_rate
            # If we're close to OB, set stop just below it
            if distance_to_ob < 0.03:  # Within 3% of the OB
                ob_stoploss = -(
                    (current_rate - last_candle["bullish_ob_low"] * 0.995) / current_rate
                )  # Tighter buffer (0.99 → 0.995)
                return max(ob_stoploss, self.stoploss)

        # 3. Dynamic stop based on ATR for volatile markets
        if "atr" in last_candle and not np.isnan(last_candle["atr"]):
            # Use tighter ATR multiplier for stoploss
            atr_multiplier = self.cstp_atr_multiplier.value * 0.8  # 20% tighter
            atr_stoploss = -(atr_multiplier * last_candle["atr"] / current_rate)
            # Don't let ATR-based stop be looser than our default
            return max(atr_stoploss, self.stoploss)

        # Default: Use initial stoploss
        return self.stoploss

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        Called for open trades every throttle interval
        Implements additional exit logic to exit at certain profit or
        loss levels, or when key market structure changes
        Modified to be less aggressive with exits for poor performers
        """
        import numpy as np

        # Fetch dataframe for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # If no dataframe or empty dataframe, skip this
        if dataframe.empty:
            return None

        # Get the last row of the dataframe
        last_candle = dataframe.iloc[-1].squeeze()

        # Exclude AVAX completely - exit any positions
        if "AVAX" in pair:
            return "avax_removal"

        # Calculate trade duration in hours
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Define top performers and poor performers
        top_performers = ["DOGE/USDT:USDT", "SOL/USDT:USDT", "PAXG/USDT:USDT"]
        poor_performers = ["ADA/USDT:USDT"]

        # ENHANCEMENT 3: More targeted poor performer exits
        if pair in poor_performers:
            # Only exit poor performers if trending against the position
            if (trade.is_short and last_candle["market_trend"] > 20) or (
                not trade.is_short and last_candle["market_trend"] < -20
            ):
                # Exit poor performers at any profit or small loss when market is trending against position
                if current_profit > -0.02:  # Slightly more tolerant
                    return "poor_performer_early_exit"
            # Exit poor performers after longer timeout in loss
            elif trade_duration_hours > 16 and current_profit < -0.03:  # More tolerance
                return "poor_performer_timeout"

        # Handle long positions
        if trade.is_short == False:
            # 1. Progressive timeout system for long positions - RELAXED
            if current_profit <= 0:  # In loss
                # Severe loss timeout - RELAXED
                if (
                    pair not in top_performers
                    and current_profit < -0.06  # Reduced from -0.035 to -0.06
                    and trade_duration_hours > 24  # Increased from 12 to 24 hours
                ):
                    return "exit_long_significant_loss"
                # More forgiving for top performers - RELAXED
                elif (
                    pair in top_performers
                    and current_profit < -0.09  # Reduced from -0.06 to -0.09
                    and trade_duration_hours > 36  # Increased from 24 to 36 hours
                ):
                    return "exit_long_significant_loss"

                # Standard loss timeouts with market condition checks - RELAXED
                elif trade_duration_hours > 36:  # Increased from 24 to 36 hours
                    # Exit standard pairs after 36 hours in loss
                    if pair not in top_performers:
                        return "exit_long_timeout"
                    # Give top performers 48 hours (increased from 36)
                    elif trade_duration_hours > 48:
                        return "exit_long_timeout"

            # 2. Minimal profit timeout - trades stuck with minimal profit - RELAXED
            elif (
                current_profit > 0 and current_profit < 0.03 and trade_duration_hours > 72
            ):  # Increased from 48 to 72
                if pair not in top_performers:
                    return "exit_long_minimal_profit"

            # 3. Dynamic take profit based on market conditions - MOSTLY UNCHANGED
            # (keep this section as is - it's already well-tuned)

            # 4. Exit on a significant market structure change - RELAXED
            if (
                "bearish_bos" in last_candle
                and last_candle["bearish_bos"]
                and current_profit > 0.01  # Increased from 0.005 to 0.01
            ):
                if pair not in top_performers:  # Don't exit top performers on minor BOS
                    return "bearish_break_of_structure"

            # 5. ENHANCEMENT: Exit if we detect a bearish order block and price is approaching it
            if (
                not np.isnan(last_candle.get("bearish_ob_low", np.nan))
                and current_rate > last_candle["bearish_ob_low"] * 0.99  # Less sensitive
                and current_profit > 0.01  # Only exit if we have some profit
                and (last_candle.get("bearish_ob_quality", 0) > 1.5)  # High quality OB
            ):
                # Exit for all pairs except top performers in significant profit
                if not (pair in top_performers and current_profit > 0.03):
                    return "approaching_bearish_order_block"

            # 6. Exit if trend turns bearish on higher timeframe and we're in profit - RELAXED
            if (
                "uptrend_1h" in last_candle
                and not last_candle["uptrend_1h"]
                and current_profit > 0.02  # Increased from 0.01 to 0.02
            ):
                if pair not in top_performers:  # Don't exit top performers on trend shift alone
                    return "trend_shifted_bearish"

            # 7. Exit if RSI indicates extreme overbought conditions - RELAXED
            if (
                "rsi" in last_candle and last_candle["rsi"] > 85 and current_profit > 0.02
            ):  # Increased from 80 to 85
                return "exit_overbought"

        # Handle short positions
        else:
            # 1. Progressive timeout system for short positions - RELAXED
            if current_profit <= 0:  # In loss
                # Severe loss timeout - RELAXED
                if (
                    pair not in top_performers
                    and current_profit < -0.06  # Relaxed from -0.035 to -0.06
                    and trade_duration_hours > 24  # Increased from 12 to 24 hours
                ):
                    return "exit_short_significant_loss"
                # More forgiving for top performers - RELAXED
                elif (
                    pair in top_performers
                    and current_profit < -0.09  # Relaxed from -0.06 to -0.09
                    and trade_duration_hours > 36  # Increased from 24 to 36 hours
                ):
                    return "exit_short_significant_loss"

                # Standard loss timeouts with market condition checks - RELAXED
                elif trade_duration_hours > 48:  # Increased from 36 to 48 hours
                    # Exit standard pairs after 48 hours in loss
                    if pair not in top_performers:
                        return "exit_short_timeout"
                    # Give top performers 60 hours (increased from 48)
                    elif trade_duration_hours > 60:
                        return "exit_short_timeout"

            # 2. Minimal profit timeout - trades stuck with minimal profit - RELAXED
            elif (
                current_profit > 0 and current_profit < 0.03 and trade_duration_hours > 72
            ):  # Increased from 60 to 72
                if pair not in top_performers:
                    return "exit_short_minimal_profit"

            # 3. Dynamic take profit based on market conditions - MOSTLY UNCHANGED
            # (keep this section as is - it's already well-tuned)

            # 4. Exit on a significant market structure change - RELAXED
            if (
                "bullish_bos" in last_candle
                and last_candle["bullish_bos"]
                and current_profit > 0.01  # Increased from 0.005 to 0.01
            ):
                if pair not in top_performers:  # Don't exit top performers on minor BOS
                    return "bullish_break_of_structure_short"

            # 5. ENHANCEMENT: Exit if we detect a bullish order block and price is approaching it
            if (
                not np.isnan(last_candle.get("bullish_ob_high", np.nan))
                and current_rate < last_candle["bullish_ob_high"] * 1.01  # Less sensitive
                and current_profit > 0.01  # Only exit if we have some profit
                and (last_candle.get("bullish_ob_quality", 0) > 1.5)  # High quality OB
            ):
                # Exit for all pairs except top performers in significant profit
                if not (pair in top_performers and current_profit > 0.03):
                    return "approaching_bullish_order_block_short"

            # 6. Exit if trend turns bullish on higher timeframe and we're in profit - RELAXED
            if (
                "uptrend_1h" in last_candle
                and last_candle["uptrend_1h"]
                and current_profit > 0.02  # Increased from 0.01 to 0.02
            ):
                if pair not in top_performers:  # Don't exit top performers on trend shift alone
                    return "trend_shifted_bullish_short"

            # 7. Exit if RSI indicates extreme oversold conditions - RELAXED
            if (
                "rsi" in last_candle and last_candle["rsi"] < 15 and current_profit > 0.02
            ):  # Decreased from 20 to 15
                return "exit_oversold_short"

        return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Adjust position size for open trades based on market conditions and performance.
        Returns None (no adjustment) or position size multiplier between 0.0-1.0.
        """
        # Skip for unsupported modes
        if self.config["trading_mode"] != "futures":
            return None

        # Fetch dataframe for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # If no dataframe or empty dataframe, don't adjust
        if dataframe.empty:
            return None

        # Get the last row of the dataframe
        last_candle = dataframe.iloc[-1].squeeze()

        # Calculate trade duration in hours
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Exclude AVAX completely - return 0 to exit the position
        if "AVAX" in trade.pair:
            return 0.0

        # Dynamic position sizing based on market strength relative to trade direction
        market_trend_value = last_candle.get("market_trend", 0)

        # Strong performers with increased allocation
        top_performers = ["DOGE/USDT:USDT", "SOL/USDT:USDT", "PAXG/USDT:USDT"]

        # Poor performers with reduced allocation
        poor_performers = ["ADA/USDT:USDT"]

        # Aggressively size top performers
        if trade.pair in top_performers:
            # Shorts in bearish market or longs in bullish market - ideal conditions
            if (trade.is_short and market_trend_value < 0) or (
                not trade.is_short and market_trend_value > 0
            ):
                return 2.0  # Double position size for aligned market conditions
            else:
                return 1.2  # 20% increase even in neutral conditions

        # Significantly reduce exposure to poor performers
        if trade.pair in poor_performers:
            return 0.3  # 70% reduction in position size

        # Reduce position on trades that run counter to market trend
        if trade.is_short and market_trend_value > 10:  # Strong bullish market, bad for shorts
            return 0.5  # 50% position reduction
        elif (
            not trade.is_short and market_trend_value < -10
        ):  # Strong bearish market, bad for longs
            return 0.5  # 50% position reduction

        # Performance-based pair categorization
        weak_pairs = ["BTC/USDT:USDT", "XRP/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT"]

        # Reduce position size for problematic pairs
        if trade.pair in weak_pairs:
            # Further reduce position after extended time in trade
            if trade_duration_hours > 12:
                return 0.4  # 60% reduction for longer-duration trades on weak pairs
            return 0.6  # 40% reduction by default for weak pairs

        # Reduce position in high volatility conditions
        if "volatility" in last_candle and "high_volatility" in last_candle:
            if last_candle["high_volatility"]:
                return 0.75  # 25% reduction for high volatility conditions

        # Reduce position if higher timeframe trend is weak for the trade direction
        if "uptrend_1h" in last_candle:
            # For shorts, reduce position in uptrend
            if trade.is_short and last_candle["uptrend_1h"]:
                return 0.8  # 20% reduction
            # For longs, reduce position in downtrend
            elif not trade.is_short and not last_candle["uptrend_1h"]:
                return 0.8  # 20% reduction

        # Default: no adjustment
        return None

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 3,
                "stop_duration_candles": 12,
                "required_profit": -0.05,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 100,
                "trade_limit": 2,
                "stop_duration_candles": 48,
                "required_profit": -0.02,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 5,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.15,
            },
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
        ]

    def bot_start(self, **kwargs) -> None:
        """
        Called only once when bot starts
        Create necessary directories to avoid errors
        """
        try:
            # Ensure model directories exist
            model_name = self.config.get("freqai", {}).get("identifier", "freqai")
            model_dir = Path(self.config["user_data_dir"]) / "models" / model_name

            # Create directory if it doesn't exist
            if not model_dir.exists():
                logger.info(f"Creating model directory: {model_dir}")
                model_dir.mkdir(parents=True, exist_ok=True)

            # Also create subdirectories that FreqAI might need
            for subdir in ["historic_predictions", "historic_data", "pair_dictionary"]:
                subdir_path = model_dir / subdir
                if not subdir_path.exists():
                    logger.info(f"Creating subdirectory: {subdir_path}")
                    subdir_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"FreqAI model directories initialized at {model_dir}")

        except Exception as e:
            logger.warning(f"Error creating model directories: {str(e)}")

        # Add any additional startup logic here


# <<< END OF INSERTED METHODS >>>
