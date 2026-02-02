# screener.py
import os
import time
import logging
import pandas as pd
from datetime import datetime, timezone
from copy import deepcopy

from bot.config import load_config, setup_logging
from bot.backtester import FastBacktester
from pybit.unified_trading import HTTP

# ===== USER SETTINGS =====
TIMEFRAMES = ["15", "60"]
DAYS_BACK = 90
INITIAL_BALANCE = 1000.0
MIN_TRADES = 5
RESULTS_DIR = "results"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# =========================

logger = logging.getLogger(__name__)

def interval_to_minutes(interval: str) -> int:
    mapping = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240, "D": 1440}
    return mapping.get(interval, 60)

def download_klines(session, symbol: str, interval: str, end_time=None, days_back=7, warmup_bars=0):
    minutes = interval_to_minutes(interval)
    candles_per_day = 1440 / minutes
    total_needed = math.ceil(days_back * candles_per_day) + warmup_bars

    if end_time:
        last_end = int(pd.Timestamp(end_time).tz_localize('UTC').timestamp() * 1000)
    else:
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
    logger.info(f"Сохранено {len(df)} свечей → {csv_path}")

    return df

def run_screener_for_symbol(cfg, session, symbol, timeframes):
    """Анализирует эффективность стратегий без подбора TP/SL"""
    results_list = []
    symbol_results_dir = os.path.join(RESULTS_DIR, symbol)
    os.makedirs(symbol_results_dir, exist_ok=True)

    # Загрузка данных
    all_timeframes = set(timeframes + ["1", "60", "240"])
    df_cache = {}
    for tf in all_timeframes:
        warmup = 55 * 60 if tf == "1" else 55
        df_cache[tf] = download_klines(session, symbol, tf, None, DAYS_BACK, warmup_bars=warmup)

    df_1m = df_cache["1"]
    df_1h = df_cache["60"]
    df_240 = df_cache["240"]

    # Тестируем каждую стратегию
    strategies = ["trend", "range", "breakout", "countertrend"]
    for tf in timeframes:
        df_main = df_cache[tf]
        for strategy_name in strategies:
            logger.info(f"[{symbol}] Тестирование стратегии: {strategy_name} на TF={tf}")

            test_cfg = deepcopy(cfg)
            test_cfg.symbol = symbol
            test_cfg.timeframe = tf
            test_cfg.forced_strategy = strategy_name  # ← Ключевое изменение

            from bot.strategy_factory import create_strategy
            strategy = create_strategy(test_cfg, df_1h)
            strategy.df_higher = df_240

            bt = FastBacktester(test_cfg, strategy, INITIAL_BALANCE, df_1h=df_1h, df_240=df_240)
            report = bt.run(df_main, df_1m)

            if report["Total Trades"] >= MIN_TRADES:
                final_increase = (report["Final Balance"] / report["Initial Balance"] - 1) * 100
                results_list.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "strategy": strategy_name,
                    "Total Trades": report["Total Trades"],
                    "Winrate %": report["Winrate %"],
                    "Profit Factor": report["Final Balance"] / max(1, report["Initial Balance"]),
                    "final balance increase %": final_increase,
                    "Max Drawdown %": report["Max Drawdown"]
                })

    if not results_list:
        logger.warning(f"[{symbol}] Ни одна стратегия не показала достаточного количества сделок.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values("final balance increase %", ascending=False).reset_index(drop=True)

    # Сохраняем лучшую стратегию
    best = results_df.iloc[0]
    optimized_config = {
        "symbol": symbol,
        "timeframe": str(best["timeframe"]),
        "forced_strategy": best["strategy"],
        "risk_percentage": 0.008,
        "leverage": 3,
        "testnet": True,
        "dry_run": True
    }
    
    optimized_path = os.path.join(symbol_results_dir, "config_optimized.yaml")
    import yaml
    with open(optimized_path, "w", encoding="utf-8") as f:
        yaml.dump(optimized_config, f, allow_unicode=True, sort_keys=False)
    logger.info(f"✅ [{symbol}] Лучшая стратегия: {best['strategy']} на TF={best['timeframe']}")

    summary_file = os.path.join(symbol_results_dir, f"screener_results_{symbol}.csv")
    results_df.to_csv(summary_file, index=False)
    logger.info(f"📊 [{symbol}] Анализ завершён → {summary_file}")

    return results_df

def run_screener():
    cfg = load_config("config.yaml")
    setup_logging(cfg.log_level)
    session = HTTP(testnet=cfg.testnet, api_key=cfg.api_key, api_secret=cfg.api_secret)

    if hasattr(cfg, 'symbols') and isinstance(cfg.symbols, list):
        symbols_to_test = [s['symbol'] for s in cfg.symbols]
    else:
        symbols_to_test = [cfg.symbol]

    all_results = []
    for symbol in symbols_to_test:
        logger.info(f"=== Начало анализа для {symbol} ===")
        results_df = run_screener_for_symbol(cfg, session, symbol, TIMEFRAMES)
        if not results_df.empty:
            all_results.append(results_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "combined_screener_results.csv"), index=False)
        logger.info("📊 Общий отчёт сохранён: results/combined_screener_results.csv")

if __name__ == "__main__":
    import math  # добавлен импорт
    run_screener()