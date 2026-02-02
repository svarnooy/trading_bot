# run_backtester.py
import os
import logging
import pandas as pd
from datetime import datetime, timezone
import math
from pybit.unified_trading import HTTP

from bot.config import load_config, setup_logging
from bot.backtester import FastBacktester
from bot.strategy_factory import create_strategy

# ===== НАСТРОЙКИ ДЛЯ 4H =====
SYMBOL = "XAUTUSDT"
TIMEFRAME = "240"  # 4 часа
DAYS_BACK = 180     # увеличено для 4H111
INITIAL_BALANCE = 200000.0
RESULTS_DIR = "results"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def interval_to_minutes(interval: str) -> int:
    mapping = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240, "D": 1440}
    return mapping.get(interval, 240)

def download_klines(session, symbol: str, interval: str, days_back=7, warmup_bars=0):
    minutes = interval_to_minutes(interval)
    candles_per_day = 1440 / minutes
    total_needed = math.ceil(days_back * candles_per_day) + warmup_bars
    last_end = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows = []
    while len(all_rows) < total_needed:
        r = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            end=last_end,
            limit=1000
        )
        result = r.get("result", {}).get("list", [])
        if not result:
            break
        all_rows.extend(result)
        last_end = min(int(x[0]) for x in result) - 1
        time.sleep(0.1)

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().set_index("timestamp").sort_index()
    if len(df) > 0:
        df = df.iloc[:-1]
    df = df.tail(total_needed)

    csv_path = os.path.join(DATA_DIR, f"{symbol}_{interval}m.csv")
    df.to_csv(csv_path)
    logging.info(f"Сохранено {len(df)} свечей → {csv_path}")
    return df

def main():
    cfg = load_config("config.yaml")
    setup_logging(cfg.log_level)
    
    session = HTTP(testnet=cfg.testnet, api_key=cfg.api_key, api_secret=cfg.api_secret)

    # Загрузка данных для 4H
    df_main = download_klines(session, SYMBOL, TIMEFRAME, DAYS_BACK, warmup_bars=200)
    df_1h = download_klines(session, SYMBOL, "60", DAYS_BACK, warmup_bars=55)
    df_240 = download_klines(session, SYMBOL, "240", DAYS_BACK, warmup_bars=55)
    # Для 4H не нужен 1m, но оставим для backtester
    df_1m = download_klines(session, SYMBOL, "1", DAYS_BACK, warmup_bars=55*60)

    # Создание стратегии
    strategy = create_strategy(cfg, df_1h)
    strategy.df_higher = df_240

    # Запуск бэктеста
    bt = FastBacktester(cfg, strategy, INITIAL_BALANCE, df_1h=df_1h, df_240=df_240)
    report = bt.run(df_main, df_1m)

    # Сохранение результата
    result_df = pd.DataFrame([report])
    result_path = os.path.join(RESULTS_DIR, f"result_{SYMBOL}_{TIMEFRAME}m.csv")
    result_df.to_csv(result_path, index=False)
    logging.info(f"Сохранен результат → {result_path}")

if __name__ == "__main__":
    import time
    main()