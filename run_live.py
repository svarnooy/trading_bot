# run_live.py
import os
import time
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

from bot.config import load_config, setup_logging
from bot.trading_bot import TradingBot
from bot.portfolio_manager import PortfolioManager
from pybit.unified_trading import HTTP

def run_bot(bot):
    """Запускает один бот в отдельном потоке"""
    try:
        bot.run()  # ← Бесконечный цикл внутри потока
    except Exception as e:
        logging.exception(f"Бот {bot.cfg.symbol} упал: {e}")

def main():
    cfg = load_config("config.yaml")
    setup_logging(cfg.log_level)
    
    session = HTTP(
        testnet=cfg.testnet,
        demo=getattr(cfg, 'demo', False),
        api_key=cfg.api_key,
        api_secret=cfg.api_secret
    )
    
    portfolio_manager = PortfolioManager(cfg)
    
    if hasattr(cfg, 'symbols') and isinstance(cfg.symbols, list):
        symbols_config = cfg.symbols
    else:
        symbols_config = [{
            'symbol': cfg.symbol,
            'timeframe': cfg.timeframe,
            'use_full_strategy': getattr(cfg, 'use_full_strategy', True)
        }]
    
    threads = []
    
    for symbol_cfg in symbols_config:
        symbol_specific_cfg = SimpleNamespace(**vars(cfg))
        symbol_specific_cfg.symbol = symbol_cfg['symbol']
        symbol_specific_cfg.timeframe = symbol_cfg['timeframe']
        symbol_specific_cfg.use_full_strategy = symbol_cfg.get('use_full_strategy', True)
        
        bot = TradingBot(symbol_specific_cfg, portfolio_manager)
        
        # ЗАПУСК В ОТДЕЛЬНОМ ПОТОКЕ ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
        t = threading.Thread(target=run_bot, args=(bot,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)  # Пауза между запусками
    
    logging.info(f"✅ Запущено {len(threads)} ботов. Ожидание сигналов...")
    
    # Основной поток не завершается
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Остановка ботов...")

if __name__ == "__main__":
    main()