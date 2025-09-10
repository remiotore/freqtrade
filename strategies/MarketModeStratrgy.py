import numpy as np
from pandas import DataFrame
from enum import Enum
from freqtrade.strategy import IStrategy

class MarketModeStrategy(IStrategy):
    class MarketMode(Enum):
        BEAR = -1
        BULL = 1
        SIDEWAYS = 0

    can_short: bool = True

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    stoploss = -0.10

    timeframe = "15m"

    startup_candle_count: int = 20

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def give_market_mode_indicator(self, dataframe: DataFrame, metadata: dict):
        mode_continue_times = 2
        mode_pairs_proportion = 0.5
        inf_timeframe = '1h'
        dataframe['market_mode'] = np.nan
        for idx, row in dataframe.iterrows():
            row_time = row['date']
            bear_count = 0
            bull_count = 0
            total_pairs = 0
            # for inf_pair in self.config['exchange']['pair_whitelist']:
            for inf_pair in [metadata['pair']]:
                pair_dataframe = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=inf_timeframe)
                if pair_dataframe is not None and len(pair_dataframe) >= mode_continue_times:
                    matching_indices = pair_dataframe[pair_dataframe['date'] <= row_time].index
                    if len(matching_indices) > 0:
                        matching_index = matching_indices[-1]
                        if matching_index >= mode_continue_times - 1:
                            total_pairs += 1
                            close = pair_dataframe['close'].iloc[matching_index - mode_continue_times + 1:matching_index + 1]
                            if mode_continue_times > 1:
                                if all(close.iloc[i] < close.iloc[i - 1] for i in range(1, mode_continue_times)):
                                    bear_count += 1
                                elif all(close.iloc[i] > close.iloc[i - 1] for i in range(1, mode_continue_times)):
                                    bull_count += 1
            if total_pairs > 0:
                bear_proportion = bear_count / total_pairs
                bull_proportion = bull_count / total_pairs
                if bear_proportion >= mode_pairs_proportion:
                    dataframe.loc[idx, 'market_mode'] = MarketModeStrategy.MarketMode.BEAR.value
                elif bull_proportion >= mode_pairs_proportion:
                    dataframe.loc[idx, 'market_mode'] = MarketModeStrategy.MarketMode.BULL.value
                else:
                    dataframe.loc[idx, 'market_mode'] = MarketModeStrategy.MarketMode.SIDEWAYS.value

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.give_market_mode_indicator(dataframe, metadata)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['market_mode'] == MarketModeStrategy.MarketMode.BULL.value
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'bull_enter')

        dataframe.loc[
            (
                dataframe['market_mode'] == MarketModeStrategy.MarketMode.BEAR.value
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'bear_enter')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['market_mode'] == MarketModeStrategy.MarketMode.BULL.value
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'bull_exit')

        dataframe.loc[
            (
                dataframe['market_mode'] == MarketModeStrategy.MarketMode.BEAR.value
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'bear_exit')

        return dataframe