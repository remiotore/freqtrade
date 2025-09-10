import psutil
import time
import os
import gc
import objgraph
import datetime
import importlib.util
import logging
from Signal import Signal
from pandas import DataFrame
import pandas as pd
from freqtrade.strategy import IStrategy, informative
from freqtrade.persistence import Trade
from datetime import datetime
import Indicators as ind



class NostalgiaForSimplicity(IStrategy):
    """
    Strategy managing multiple signals with priority
    """

    INTERFACE_VERSION = 3

    def __init__(self, config: dict) -> None:
        self.log = logging.getLogger(__name__)
        self.log.info("INICIANDO...")
        self.indicators = self.load("indicators", Signal)
        self.log.info(f"Indicadores cargados: {[type(ind).__name__ for ind in self.indicators]}")
        self.signals = self.load("signals", Signal)
        super().__init__(config)
        


    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called once before the bot starts. Checks for modified signals and indicators and reloads if needed.
        """
        # Verificar si es la primera vez que se ejecuta
        if not hasattr(self, '_has_run_once'):
            self._has_run_once = False

        # Inicializar almacenamiento para archivos monitoreados
        if not hasattr(self, "_last_loaded_files"):
            self._last_loaded_files = {}

        # Verificar cambios en archivos
        files_modified = self._check_for_file_changes()

        # En la primera iteración, simplemente marcar como completada y salir
        if not self._has_run_once:
            self.log.info("Primera ejecución completada: las marcas de tiempo han sido registradas.")
            self._has_run_once = True
            return

        # Recargar señales e indicadores solo si se detectaron cambios
        if files_modified.get("signals"):
            self.log.info("Cambios detectados en 'signals'. Recargando...")
            self._reload_component("signals", Signal)
            self._refresh_signals()

        if files_modified.get("indicators"):
            self.log.info("Cambios detectados en 'indicators'. Recargando...")
            self._reload_component("indicators", Signal)
            self._reload_component("signals", Signal)
            self._refresh_signals()

        self.log_memory_usage() 

            
    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        self.log.info(f"Used memory: {mem_info.rss / 1024 ** 2:.2f} MB")
        #self.debug_memory_leak()
        #self.check_dataframe_leaks()
        #self.trace_dataframe_references()
        gc.collect()  # Forzar recolección de basura

    def debug_memory_leak(self):
        self.log.info("🔍 Objects in memory:")
        objgraph.show_growth(limit=10)  # Muestra los 10 objetos que más han crecido

        # Si sospechamos que hay demasiados objetos del mismo tipo:
        objgraph.show_most_common_types(limit=10)

    def check_dataframe_leaks(self):
        dataframes = [obj for obj in gc.get_objects() if isinstance(obj, pd.DataFrame)]
        print(f"⚠️ Hay {len(dataframes)} DataFrames en memoria")

        for i, df in enumerate(dataframes[:5]):  # Solo mostramos los primeros 5
            self.log.info(f"📊 DataFrame {i}: {df.shape}, Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
   

    def trace_dataframe_references(self):
        dataframes = [obj for obj in gc.get_objects() if isinstance(obj, pd.DataFrame)]
        
        if not dataframes:
            self.log.info("✅ No hay DataFrames retenidos en memoria.")
            return

        self.log.info("🔍 Mostrando referencias a un DataFrame sospechoso...")
        objgraph.show_backrefs(dataframes[-1], max_depth=3, filename="dataframe_refs.png")

    def _check_for_file_changes(self) -> dict:
        """
        Verifica los cambios en los archivos de las carpetas "signals" y "indicators".
        """
        directories = [
            ("indicators", "indicators"),
            ("signals", "signals"),
        ]
        files_modified = {"signals": False, "indicators": False}

        for dir_name, attr_name in directories:
            dir_path = os.path.join(os.path.dirname(__file__), dir_name)

            for file in os.listdir(dir_path):
                if file.endswith(".py") and file != "__init__.py":
                    file_path = os.path.join(dir_path, file)

                    # Obtener el tiempo de modificación del archivo
                    last_modified_time = os.path.getmtime(file_path)
                    self.log.debug(f"Archivo: {file}, Marca de tiempo actual: {last_modified_time}")

                    # Si es la primera iteración, solo registrar las fechas
                    if not self._has_run_once:
                        self._last_loaded_files[file_path] = last_modified_time
                        self.log.debug(f"Primera ejecución: registrada marca de tiempo para {file}.")
                    else:
                        # Detectar cambios en iteraciones posteriores
                        prev_time = self._last_loaded_files.get(file_path)
                        if prev_time != last_modified_time:
                            self.log.debug(f"Cambio detectado en {file}: {prev_time} -> {last_modified_time}")
                            self._last_loaded_files[file_path] = last_modified_time
                            files_modified[attr_name] = True

        return files_modified

    def _reload_component(self, component_name: str, loader) -> None:
        """
        Recarga un componente especificado (signals o indicators).
        """
        self.log.debug(f"Recargando {component_name}...")
        setattr(self, component_name, self.load(component_name, loader))
        self.log.debug(f"{component_name.capitalize()} recargados.")

    def _refresh_signals(self) -> None:
        """
        Fuerza la actualización de los pares para regenerar señales de entrada y salida.
        """
        current_pairs = self.dp.current_whitelist() if callable(self.dp.current_whitelist) else self.dp.current_whitelist
        self.process_only_new_candles = False
        self.analyze(current_pairs)
        self.process_only_new_candles = True


    def load(self, signal_dir="signals", base_class=Signal):
        """
        Carga dinámicamente clases desde la carpeta dada y las retorna en una lista ordenada por prioridad.
        """
        components = []
        signal_dir = os.path.join(os.path.dirname(__file__), signal_dir)

        if not os.path.exists(signal_dir):
            self.log.warning(f"El directorio {signal_dir} no existe.")
            return components

        for file in os.listdir(signal_dir):
            if file.endswith(".py") and file != "__init__.py":
                module_name = file[:-3]
                file_path = os.path.join(signal_dir, file)

                # Dinámicamente importar el módulo
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Buscar subclases del tipo base_class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, base_class) and attr is not base_class:
                        components.append(attr(self))
                        self.log.debug(f"Clase cargada: {attr.__name__} desde {file_path}")

        # Ordenar las clases por prioridad si aplicable
        return sorted(components, key=lambda component: component.get_priority())



    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
            'EMA_12': {'color': 'red'},
            'EMA_26': {'color': 'blue'},
            'EMA_50': {'color': 'green'},
            'EMA_200': {'color': 'yellow'},
        }
        plot_config['subplots'] = {
            # Create subplot MACD
            "downtrend": {
                'is_downtrend': {'color': 'red'},
            },
            "downtrend_signals": {
                'downtrend_signals': { 'color': 'blue'},
            },
            "ADX": {
                'ADX': { 'color': 'green'},
            }
        }

        return plot_config


    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:   
        for indicator in self.indicators:
            if indicator.enabled:
                self.log.debug(f"Populating indicators for {indicator.get_signal_tag()}.")
                df = indicator.populate_indicators(df)  # Llamar al método de cada señal
        return df
    

    @informative('15m')
    def populate_indicators_15m(self, df: DataFrame, metadata: dict) -> DataFrame:
        for indicator in self.indicators:
            if indicator.enabled:
                self.log.debug(f"Populating indicators 15m for {indicator.get_signal_tag()}.")
                df = indicator.populate_indicators_15m(df)  # Llamar al método de cada señal
        return df


    @informative('1h')
    def populate_indicators_1h(self, df: DataFrame, metadata: dict) -> DataFrame:
        for indicator in self.indicators:
            if indicator.enabled:
                self.log.debug(f"Populating indicators 1h for {indicator.get_signal_tag()}.")
                df = indicator.populate_indicators_1h(df)  # Llamar al método de cada señal
        return df


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on plugin logic, with custom tags for each signal,
        while avoiding entries during a downtrend.
        """
        # Verificar que la vela actual tiene datos válidos
        if dataframe.empty or dataframe.iloc[-1].isna().any():
            self.log.warning(f"Skipping populate_entry_trend for {metadata.get('pair')} due to missing candle data.")
            return dataframe

        # Ensure required columns exist
        if "enter_long" not in dataframe.columns:
            dataframe["enter_long"] = 0
        if "enter_tag" not in dataframe.columns:
            dataframe["enter_tag"] = None  # Initialize with None

        pair = metadata.get("pair", "Unknown")  # Retrieve the trading pair from metadata

        for signal in self.signals:
            if not signal.enabled:
                continue

            self.log.debug(f"Checking entry signals from plugin {signal.get_signal_tag()}.")

            # Create a lazy evaluation function for the signal
            def lazy_evaluation():
                entry_signal = signal.entry_signal(dataframe, metadata)

                # Ensure entry_signal is a boolean Series
                if not isinstance(entry_signal, pd.Series) or entry_signal.dtype != bool:
                    raise TypeError(f"Signal {signal.get_signal_tag()} returned an invalid entry signal type.")

                return entry_signal

            # Generate an initial mask to evaluate signals
            new_signals = (dataframe["enter_long"] == 0)

            # Apply signals lazily
            if new_signals.any():
                entry_signal = lazy_evaluation()  # Evaluate the signal only if necessary
                final_signals = new_signals & entry_signal

                # Assign 1 to 'enter_long' for active signals
                dataframe.loc[final_signals, "enter_long"] = 1

                # Assign tags with the prefix 'enter_' to 'enter_tag'
                dataframe.loc[final_signals, "enter_tag"] = f"enter_{signal.get_signal_tag()}"

                # Log the pair and the generated signals
                signal_count = final_signals.sum()
                if signal_count > 0:
                    self.log.info(f"Signal {signal.get_signal_tag()} generated {signal_count} entry signal(s) for pair {pair}.")

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on plugin logic, with custom tags for each signal.
        """
        # Verificar que la vela actual tiene datos válidos
        if dataframe.empty or dataframe.iloc[-1].isna().any():
            self.log.warning(f"Skipping populate_exit_trend for {metadata.get('pair')} due to missing candle data.")
            return dataframe

        # Asegurar columnas necesarias
        if "exit_long" not in dataframe.columns:
            dataframe["exit_long"] = 0
        if "exit_tag" not in dataframe.columns:
            dataframe["exit_tag"] = ""

        pair = metadata.get("pair", "Unknown")  # Obtener el par desde metadata

        for signal in self.signals:
            if not signal.enabled:
                #self.log.info(f"Plugin {plugin.get_plugin_tag()} is disabled, skipping exit signal.")
                continue

            self.log.debug(f"Checking exit signals from plugin {signal.get_signal_tag()}.")
            exit_signal = signal.exit_signal(dataframe, metadata)

            # Verificar que exit_signal sea una Series booleana
            if not isinstance(exit_signal, pd.Series) or exit_signal.dtype != bool:
                raise TypeError(f"Signal {signal.get_signal_tag()} returned an invalid exit signal type.")

            # Aplicar señales de salida al DataFrame
            dataframe.loc[exit_signal, "exit_long"] = 1

            # Asignar etiquetas (exit_tag) para las señales activas
            dataframe.loc[exit_signal, "exit_tag"] = f"exit_{signal.get_signal_tag()}"


            # Registrar el par y las señales generadas
            signal_count = exit_signal.sum()
            if signal_count > 0:
                self.log.info(f"Signal {signal.get_signal_tag()} generated {signal_count} exit signal(s) for pair {pair}.")

        return dataframe


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):

        params = {
            "pair": pair,
            "strategy": self,
            "trade": trade,
            "current_time": current_time,
            "current_rate": current_rate,
            "current_profit": current_profit,
            **kwargs,
        }

        exit_signal = False  # Valor por defecto si ninguna señal se activa

        for signal in self.signals:
            if signal.enabled:
                signal_triggered = signal.custom_exit(**params)  # Llamada al método de la señal

                if isinstance(signal_triggered, bool) and signal_triggered:
                    return True  # Prioridad: si alguna señal devuelve True, se retorna inmediatamente
                elif isinstance(signal_triggered, str) and signal_triggered.strip():
                    exit_signal = signal_triggered  # Guardar la cadena no vacía, pero seguir buscando True

        return exit_signal  # Devuelve la última cadena no vacía encontrada o False si no hubo ninguna
    
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, 
                        current_profit: float, **kwargs) -> float:

        params = {
            "pair": pair,
            "trade": trade,
            "current_time": current_time,
            "current_rate": current_rate,
            "current_profit": current_profit,
            **kwargs,
        }

        stoploss_value = None  # Por defecto, usa el stop-loss de la estrategia

        for signal in self.signals:
            if signal.enabled:
                signal_stoploss = signal.custom_stoploss(**params)  # Llamada al método de la señal

                if isinstance(signal_stoploss, float):  # Si la señal devuelve un stop-loss válido
                    if stoploss_value is None or signal_stoploss > stoploss_value:
                        stoploss_value = signal_stoploss  # Usa el stop-loss menos restrictivo (más alto)

        return stoploss_value  # Devuelve el stop-loss más alto encontrado o None si no hay cambios
    
    
    def adjust_trade_position(
        self,
        trade: Trade,
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
    ) -> float | None | tuple[float | None, str | None]:
        """
        Permite ajustar la posición de un trade en curso.
        Se invoca a las señales configuradas y, si alguna devuelve un valor, se utiliza ese valor para ajustar la posición.
        """
        # Empaquetar todos los parámetros en un diccionario
        params = {
            "trade": trade,
            "current_time": current_time,
            "current_rate": current_rate,
            "current_profit": current_profit,
            "min_stake": min_stake,
            "max_stake": max_stake,
            "current_entry_rate": current_entry_rate,
            "current_exit_rate": current_exit_rate,
            "current_entry_profit": current_entry_profit,
            "current_exit_profit": current_exit_profit,
            **kwargs,
        }

        adjusted_position = None
        for signal in self.signals:
            if signal.enabled:
                # Llamamos al método adjust_trade_position de la señal pasándole todos los parámetros
                result = signal.adjust_trade_position(**params)
                if result is not None:
                    adjusted_position = result
                    self.log.info(f"Signal {signal.get_signal_tag()} ajustó la posición del trade a: {result}")
                    # Si una señal ya ajustó la posición, salimos del bucle
                    break
        return adjusted_position

