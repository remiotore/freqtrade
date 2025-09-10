import logging
from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IntParameter, IStrategy, DecimalParameter


logger = logging.getLogger(__name__)

class MarkowitzPortfolioStrategy(IStrategy):
    """
    Strategy implementing Modern Portfolio Theory (Markowitz), Long only.
    """

    INTERFACE_VERSION = 3

    position_adjustment_enable = True

    minimal_roi = {
        "0": 1  # No ROI required as we're using portfolio optimization
    }

    stoploss = -0.5  # No stoploss as we're using portfolio optimization

    trailing_stop = False

    can_short = False

    timeframe = "1d"

    # Timeframe for covariance estimation
    cov_timeframe = "1h"

    # Lookback period for calculating returns and covariance (in hours)
    lookback_period = IntParameter(24, 168, default=72, space="buy")  # 1-7 days in hours

    # Risk-free rate (annualized)
    risk_free_rate = 0.02

    # Risk aversion parameter
    risk_aversion = IntParameter(1, 5, default=1, space="buy")
    gamma = DecimalParameter(0.1, 1.0, default=0.8, space="buy")  # Risk aversion coefficient

    # Minimum position adjustment threshold (in percentage)
    min_adjustment_threshold = 0.05  # 5% change in weight triggers rebalancing

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate daily returns for each pair
        """
        # Calculate returns
        dataframe["returns"] = dataframe["close"].pct_change()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Always return True as we want to be in the market at all times
        """
        dataframe["enter_long"] = 1
        # dataframe["enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Always return False as we don't want to exit based on signals
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def _calculate_portfolio_weights(self, pair: str) -> tuple[float, bool]:
        """
        Calculate portfolio weights using 1h data for better covariance estimation
        Returns: (weight, success)
        """
        # Get all available pairs
        pairs = self.dp.current_whitelist()

        # Get historical data for all pairs using 1h timeframe
        lookback = self.lookback_period.value
        returns_data = {}

        for p in pairs:
            df = self.dp.get_pair_dataframe(p, self.cov_timeframe)
            if len(df) >= lookback:
                # Calculate hourly returns
                returns_data[p] = df["close"].pct_change().tail(lookback).values

        if not returns_data:
            return 0.0, False

        # Convert returns to numpy array
        returns_matrix = np.array([returns_data[p] for p in pairs])

        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)

        try:
            # Calculate optimal weights using inverse covariance matrix
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(pairs))

            # Calculate optimal weights
            numerator = inv_cov @ (
                expected_returns - self.risk_free_rate / (252 * 24) + self.gamma.value * ones
            )
            denominator = ones @ numerator
            weights = numerator / denominator

            # Apply risk aversion
            weights = weights / self.risk_aversion.value

            # Get the weight for the current pair
            pair_idx = pairs.index(pair)
            weight = weights[pair_idx]

            return weight, True

        except np.linalg.LinAlgError:
            return 0.0, False

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Calculate initial position size based on Markowitz portfolio weights
        """
        weight, success = self._calculate_portfolio_weights(pair)
        if success:
            logger.info(f"Weight for {pair}: {weight}")

        if not success:
            return 0

        # Calculate position size
        total_stake = self.wallets.get_total_stake_amount()
        position_size = abs(weight) * total_stake

        # Ensure position size is within limits
        position_size = max(min_stake, min(position_size, max_stake))

        # If weight is negative, we're shorting
        if weight < 0:
            position_size = -position_size

        return position_size

    def adjust_trade_position(
        self,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None:
        """
        Adjust position size based on updated portfolio weights
        Returns the new stake amount if adjustment is needed, None otherwise
        """
        pair = trade.pair
        weight, success = self._calculate_portfolio_weights(pair)

        if success:
            logger.info(f"Weight for {pair}: {weight}")

        if not success:
            return None

        # Calculate target position size
        total_stake = self.wallets.get_total_stake_amount()
        target_position = abs(weight) * total_stake
        target_position = max(min_stake, min(target_position, max_stake))

        if weight < 0:
            target_position = -target_position

        # Get current position size
        current_position = trade.amount * current_rate

        # Calculate position difference
        position_diff = target_position - current_position
        position_diff_pct = abs(position_diff / current_position) if current_position != 0 else 1.0

        # Only adjust if the difference is significant
        if position_diff_pct >= self.min_adjustment_threshold:
            # logger.info(
            #     f"Adjusting position for {pair}: "
            #     f"Current: {current_position:.2f}, Target: {target_position:.2f}, "
            #     f"Diff: {position_diff:.2f} ({position_diff_pct:.1%})"
            # )
            if position_diff > 0:
                try:
                    return position_diff
                except Exception as e:
                    logger.error(f"Error adjusting position for {pair}: {e}")
                    return None
            else:
                return position_diff
        else:
            # logger.debug(
            #     f"No adjustment needed for {pair}: "
            #     f"Current: {current_position:.2f}, Target: {target_position:.2f}, "
            #     f"Diff: {position_diff:.2f} ({position_diff_pct:.1%})"
            # )
            return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Set leverage to 1 as we're handling position sizing in custom_stake_amount
        """
        return 1.0
