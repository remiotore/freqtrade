

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy


class rsibb(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    minimal_roi = {
    }

    """ Iniciaremos com stop loss infinito porque
    ainda precisamos encontrar um stop adequado
    e para isso precisamos executar testes
    -0.99 = stop loss infinit0""" 
    stoploss = -0.99

    ticker_interval = '5m'

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['rsi'] = ta.RSI(dataframe)



        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column

        Lógica do RSI: não queremos comprar um ativo sobrevendido (RSI < 30), pois, 
        indica que ninguem quer e pode ficar a este nivel de preço baixo por muito tempo
        Lógica da BB: o preço tende a voltar para a média, então, se for < que a banda inferior,
        acreditamos que vai subir, em direção à média.  
        """
        dataframe.loc[
            (

                (dataframe['rsi'] > 30) &
                (dataframe["close"] < dataframe['bb_lowerband'] )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column

        Lógica do RSI: não queremos comprar um ativo sobrecomprado (RSI < 83), pois, 
        indica que está muito caro, pode não subir mais. Cair é mais provável
        Lógica da BB: o preço tende a voltar para a média, então, se for > que a banda superior,
        acreditamos que vai cair, em direção à média.  
        """

        dataframe.loc[
            (
                (dataframe["close"] > dataframe['bb_middleband'] )
            ),
            'sell'] = 1

        return dataframe