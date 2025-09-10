class RingRong(IStrategy):
    timeframe = "5m"
    can_short: bool = True

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    startup_candle_count = 200

    def informative_pairs(self):
        return [("BTC/USDT", "5m"), ("ETH/USDT", "5m")]

    def populate_indicators(self, dataframe: DataFrame, meta dict) -> DataFrame:
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        dataframe["short"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["long"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe)

        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        for pair, timeframe in self.informative_pairs():
            informative = self.dp.get_pair_dataframe(pair, timeframe)
            informative["ema"] = ta.EMA(informative, timeperiod=50)
            informative["sma_200"] = ta.SMA(informative, timeperiod=200)

            dataframe[f"trend_{pair}"] = (
                informative["close"] > informative["sma_200"]
            )
            dataframe[f"trend_strength_{pair}"] = (
                (informative["close"] > informative["ema"])
                & (informative["ema"] > informative["sma_200"])
            )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, meta dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["adx"] > 25)
                & (dataframe["short"] > dataframe["long"])
                & (dataframe["close"] > dataframe["sma_200"])
                & (dataframe["close"] <= dataframe['bb_lowerband'])
                & (dataframe["rsi"] < 30)
                & (dataframe["mfi"] < 20)
                & (dataframe["trend_BTC/USDT"] == True)
                & (dataframe["trend_strength_BTC/USDT"] == True)
                & (dataframe["trend_ETH/USDT"] == True)
                & (dataframe["trend_strength_ETH/USDT"] == True)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "adx_cross_bullish")

        dataframe.loc[
            (
                (dataframe["adx"] > 25)
                & (dataframe["short"] < dataframe["long"])
                & (dataframe["close"] < dataframe["sma_200"])
                & (dataframe["close"] >= dataframe['bb_upperband'])
                & (dataframe["rsi"] > 70)
                & (dataframe["mfi"] > 80)
                & (dataframe["trend_BTC/USDT"] == False)
                & (dataframe["trend_ETH/USDT"] == False)
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "adx_cross_bearish")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["adx"] < 25)
                | (qtpylib.crossed_below(dataframe["short"], dataframe["long"]))
                | (dataframe["close"] < dataframe["sma_200"])
                | (dataframe["close"] > dataframe['bb_middleband'])
                | (dataframe["rsi"] > 50)
                | (dataframe["mfi"] > 50)
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "exit_long")

        dataframe.loc[
            (
                (dataframe["adx"] < 25)
                | (qtpylib.crossed_above(dataframe["short"], dataframe["long"]))
                | (dataframe["close"] > dataframe["sma_200"])
                | (dataframe["close"] < dataframe['bb_middleband'])
                | (dataframe["rsi"] < 50)
                | (dataframe["mfi"] < 50)  
                & (dataframe["volume"] > 0)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "exit_short")

        return dataframe
