import arrow
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_seconds
from datetime import timedelta

pd.set_option("display.precision", 10)

class Renco1(IStrategy):
    stoploss = -0.99
    can_short = True

    trailing_stop = True
    trailing_stop_positive = 0.5
    trailing_stop_positive_offset = 0.6
    trailing_only_offset_is_reached = True

    disable_dataframe_checks = True

    atr_period = 10
    timeframe = '15m'
    timeframe_seconds = timeframe_to_seconds(timeframe)

    plot_config = {
        'main_plot': {
            'renko_close': {'color': 'blue'},
            'renko_open': {'color': 'orange'},
            'sar': {'color': 'green'},
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=50)
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=self.atr_period)

        renko_columns = [
            'date', 'renko_open', 'renko_high', 'renko_low', 'renko_close',
            'trend', 'prev_trend', 'prev2_trend', 'prev3_trend',
            'prev_date', 'new_brick'
        ]

        DATE_IDX = 0
        CLOSE_IDX = 4
        TREND_IDX = 5
        PREV_TREND_IDX = 6
        PREV2_TREND_IDX = 7
        NEW_BRICK = True
        COPIED_BRICK = False

        brick_size = np.NaN
        data = []
        prev_brick = None
        prev2_trend = False
        prev3_trend = False

        for row in dataframe.itertuples():
            if np.isnan(row.ATR):
                continue
            else:
                brick_size = row.ATR

            close = row.close
            date = row.date

            if prev_brick is None:
                trend = True
                prev_brick = [
                    date,
                    close - brick_size,
                    close,
                    close - brick_size,
                    close,
                    trend,
                    False,
                    False,
                    False,
                    date,
                    NEW_BRICK,
                ]
                prev2_trend = prev_brick[PREV_TREND_IDX]
                prev3_trend = prev_brick[PREV2_TREND_IDX]
                data.append(prev_brick)
                continue

            prev_date = prev_brick[DATE_IDX]
            prev_close = prev_brick[CLOSE_IDX]
            prev_trend = prev_brick[TREND_IDX]

            new_brick = None
            trend = prev_trend

            bricks = int(np.nan_to_num((close - prev_close) / brick_size))

            if trend and bricks >= 1:
                new_brick = [
                    date,
                    prev_close,
                    prev_close + bricks * brick_size,
                    prev_close,
                    prev_close + bricks * brick_size,
                    trend,
                    prev_trend,
                    prev2_trend,
                    prev3_trend,
                    prev_date,
                    NEW_BRICK,
                ]

            elif trend and bricks <= -2:
                trend = not trend
                new_brick = [
                    date,
                    prev_close - brick_size,
                    prev_close - brick_size,
                    prev_close - abs(bricks) * brick_size,
                    prev_close - abs(bricks) * brick_size,
                    trend,
                    prev_trend,
                    prev2_trend,
                    prev3_trend,
                    prev_date,
                    NEW_BRICK,
                ]

            elif not trend and bricks <= -1:
                new_brick = [
                    date,
                    prev_close,
                    prev_close,
                    prev_close - abs(bricks) * brick_size,
                    prev_close - abs(bricks) * brick_size,
                    trend,
                    prev_trend,
                    prev2_trend,
                    prev3_trend,
                    prev_date,
                    NEW_BRICK,
                ]

            elif not trend and bricks >= 2:
                trend = not trend
                new_brick = [
                    date,
                    prev_close + brick_size,
                    prev_close + bricks * brick_size,
                    prev_close + brick_size,
                    prev_close + bricks * brick_size,
                    trend,
                    prev_trend,
                    prev2_trend,
                    prev3_trend,
                    prev_date,
                    NEW_BRICK,
                ]

            else:
                data.append([
                    date,
                    prev_brick[1],
                    prev_brick[2],
                    prev_brick[3],
                    prev_brick[4],
                    prev_brick[5],
                    prev_brick[6],
                    prev_brick[7],
                    prev_brick[8],
                    prev_brick[9],
                    COPIED_BRICK,
                ])

            if new_brick is not None:
                data.append(new_brick)
                prev2_trend = prev_brick[PREV_TREND_IDX]
                prev3_trend = prev_brick[PREV2_TREND_IDX]
                prev_brick = new_brick

        renko_chart = pd.DataFrame(data=data, columns=renko_columns)

        merged_dataframe = dataframe.merge(renko_chart, how='left', on='date')

        merged_dataframe[['renko_open', 'renko_high', 'renko_low', 'renko_close', 'trend', 'prev_trend', 'prev2_trend', 'prev3_trend']] = merged_dataframe[
            ['renko_open', 'renko_high', 'renko_low', 'renko_close', 'trend', 'prev_trend', 'prev2_trend', 'prev3_trend']
        ].ffill()

        return merged_dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Вхід у довгу позицію на першій стрілочці PSAR або Renko
        """
        dataframe['enter_long'] = 0
        dataframe['psar_shifted'] = dataframe['sar'].shift(1)

        dataframe.loc[
            (
                (dataframe['psar_shifted'] > dataframe['close']) &  # PSAR був вище ціни
                (dataframe['sar'] < dataframe['close']) &            # PSAR тепер нижче ціни
                (dataframe['trend'] == True) &                       # Поточний brick бичачий
                (dataframe['prev_trend'] == False)                   # Попередній brick ведмежий
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Вихід з довгої позиції або вхід у коротку позицію на першій стрілочці PSAR або Renko
        """
        dataframe['enter_short'] = 0
        dataframe['exit_long'] = 0
        dataframe['psar_shifted'] = dataframe['sar'].shift(1)

        dataframe.loc[
            (
                (dataframe['psar_shifted'] < dataframe['close']) &  # PSAR був нижче ціни
                (dataframe['sar'] > dataframe['close']) &             # PSAR тепер вище ціни
                (dataframe['trend'] == False) &                      # Поточний brick ведмежий
                (dataframe['prev_trend'] == True)                    # Попередній brick бичачий
            ),
            'enter_short'
        ] = 1

        return dataframe
