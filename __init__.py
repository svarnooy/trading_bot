# bot/__init__.py

from .config import ConfigModel, load_config, setup_logging
from .trading_bot import TradingBot
from .notifier import TelegramNotifier
from .backtester import FastBacktester
from .strategy_factory import create_strategy
from .market_regime_detector import detect_market_regime

__version__ = "1.0.0"
__author__ = "Andrey"

# Убедимся, что подпапка strategies тоже является модулем
from . import strategies