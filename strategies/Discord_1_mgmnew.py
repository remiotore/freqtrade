Traceback (most recent call last):
  File "/home/pj/freqtrade/freqtrade/main.py", line 37, in main
    return_code = args['func'](args)
  File "/home/pj/freqtrade/freqtrade/commands/optimize_commands.py", line 85, in start_hyperopt
    hyperopt = Hyperopt(config)
  File "/home/pj/freqtrade/freqtrade/optimize/hyperopt.py", line 76, in __init__
    self.backtesting = Backtesting(self.config)
  File "/home/pj/freqtrade/freqtrade/optimize/backtesting.py", line 77, in __init__
    self.strategylist.append(StrategyResolver.load_strategy(self.config))
  File "/home/pj/freqtrade/freqtrade/resolvers/strategy_resolver.py", line 45, in load_strategy
    strategy: IStrategy = StrategyResolver._load_strategy(
  File "/home/pj/freqtrade/freqtrade/resolvers/strategy_resolver.py", line 190, in _load_strategy
    strategy = StrategyResolver._load_object(paths=abs_paths,
  File "/home/pj/freqtrade/freqtrade/resolvers/iresolver.py", line 114, in _load_object
    (module, module_path) = cls._search_object(directory=_path,
  File "/home/pj/freqtrade/freqtrade/resolvers/iresolver.py", line 96, in _search_object
    obj = next(cls._get_valid_object(module_path, object_name), None)
  File "/home/pj/freqtrade/freqtrade/resolvers/iresolver.py", line 63, in _get_valid_object
    spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints
  File "<frozen importlib._bootstrap_external>", line 783, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/pj/freqtrade/user_data/strategies/mgmnew.py", line 77, in <module>
    class mgmnew(IStrategy):
  File "/home/pj/freqtrade/user_data/strategies/mgmnew.py", line 118, in mgmnew
    json_data = json.load(file_object)
  File "/usr/lib/python3.8/json/__init__.py", line 293, in load
    return loads(fp.read(),
  File "/usr/lib/python3.8/json/__init__.py", line 357, in loads
    return _default_decoder.decode(s)
  File "/usr/lib/python3.8/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/usr/lib/python3.8/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 103 column 9 (char 2874)
