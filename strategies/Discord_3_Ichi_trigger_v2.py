###ERROR
`KeyError: 'tenkan_sen_1h'
2021-10-10 17:37:38,848 - freqtrade.strategy.interface - WARNING - Unable to analyze candle (OHLCV) data for pair MTL/BTC: 'tenkan_sen_1h'`

###CODE
`class Ichi_trigger_v2(IStrategy):

    timeframe = '5m'
    informative_timeframe = '1h'

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def do_indicators(self, dataframe: DataFrame, metadata: dict):
        displacement = 26
        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=9, base_line_periods=26, laggin_span=52, displacement=displacement)
        # cross indicators
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value in ('backtest', 'hyperopt'):
            assert (timeframe_to_minutes(self.timeframe) <= 5), "Backtest this strategy in 5m or 1m timeframe. Read comments for details."

        if self.timeframe == self.informative_timeframe:
            dataframe = self.do_indicators(dataframe, metadata)
        else:
            if not self.dp:
                return dataframe

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
                (dataframe['bs_pic_sig'] > dataframe['pic_l_s_trig'])
        ,
        'buy'] , 1

        # ERROR HERE
        print(dataframe['tenkan_sen_1h'])
        print(dataframe['tenkan_sen'])

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
                    (dataframe['bs_pic_sig'] < dataframe['pic_l_s_trig'])
        ,
        'sell'] = 1

        return dataframe

`

whats wrong?! :-(