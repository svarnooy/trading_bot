# backtester.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ← КРИТИЧЕСКИ ВАЖНО для мультипотока
import matplotlib.pyplot as plt
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class FastBacktester:
    def __init__(self, cfg, strategy, initial_balance: float = 1000.0, df_1h: Optional[pd.DataFrame] = None, df_240: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve = []
        self.equity_timestamps = []
        self.trades = []
        self.data_dir = "data"
        self.results_dir = "results"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Кэш данных для режимов
        self._df_1h_cache = df_1h
        self._df_240_cache = df_240
        self._last_strategy_update_time = None

    def _apply_slippage(self, price: float, side: str, atr_val: float) -> float:
        """Моделирует проскальзывание на основе ATR"""
        slippage_pct = min(0.001, atr_val * 0.1 / price)  # максимум 0.1%
        if side == "long":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def _apply_fee(self, qty_usd: float) -> float:
        """Применяет комиссию (вход + выход)"""
        fee_rate = getattr(self.cfg, "fee", 0.0011)
        return qty_usd * (2 * fee_rate)

    def run(self, df_main, df_m1):
        # СБРОС ВСЕХ СОСТОЯНИЙ
        self.balance = self.initial_balance
        self.equity_curve = []
        self.equity_timestamps = []
        self.trades = []

        logger.info("=== Запуск бэктеста ===")
        position = None
        entry_price = None
        tp = sl = None
        open_index = None
        qty_usd = qty_asset = None
        balance_before = None

        warmup_bars = getattr(self.cfg, "warmup_bars", 50)
        df_cache = df_main.iloc[:warmup_bars].copy()

        start_of_day = df_main.iloc[warmup_bars].name.date() if len(df_main) > warmup_bars else df_main.iloc[0].name.date()
        day_start_balance = self.balance
        max_daily_dd = getattr(self.cfg, "max_daily_drawdown_pct", 2.0)
        skip_day = False

        for i in range(warmup_bars, len(df_main)):
            if position is not None and open_index is None:
                logger.warning("Сброс некорректного состояния: position установлена, но open_index = None")
                position = None

            current_time = df_main.iloc[i].name

            # === ДИНАМИЧЕСКИЙ ВЫБОР СТРАТЕГИИ (раз в 4 часа) ===
            should_update_strategy = (
                self._last_strategy_update_time is None or
                (current_time - self._last_strategy_update_time).total_seconds() >= 4 * 3600
            )

            if should_update_strategy and self._df_1h_cache is not None:
                df_1h_slice = self._df_1h_cache[self._df_1h_cache.index <= current_time]
                if len(df_1h_slice) > 50:
                    from .strategy_factory import create_strategy
                    self.strategy = create_strategy(self.cfg, df_1h_slice)
                    if self._df_240_cache is not None:
                        self.strategy.df_higher = self._df_240_cache
                    self._last_strategy_update_time = current_time

            df_cache_for_signal = df_cache.copy()
            try:
                signal_series = self.strategy.generate_signals(df_cache_for_signal)
                signal = signal_series[-1] if isinstance(signal_series, (list, np.ndarray, pd.Series)) and len(signal_series) > 0 else 0
            except Exception as e:
                logging.exception(f"Ошибка генерации сигнала на i={i}: {e}")
                new_row = df_main.iloc[i:i + 1]
                df_cache = pd.concat([df_cache, new_row]).iloc[-warmup_bars:]
                continue

            if position is not None:
                try:
                    open_ts = df_main.iloc[int(open_index)].name
                    t2 = df_main.iloc[i].name
                except Exception as e:
                    logger.error(f"Ошибка получения временных меток: {e}. Сбрасываем позицию.")
                    position = None
                    open_index = None
                    continue

                next_minute = open_ts + pd.Timedelta(minutes=1)
                minute_slice = df_m1[(df_m1.index >= next_minute) & (df_m1.index <= t2)]

                exit_price = None
                result_type = None
                close_ts = None

                for ts, mrow in minute_slice.iterrows():
                    if position == "long":
                        if mrow["high"] >= tp:
                            exit_price = tp
                            result_type = "TP"
                            close_ts = ts
                            break
                        elif mrow["low"] <= sl:
                            exit_price = sl
                            result_type = "SL"
                            close_ts = ts
                            break
                    else:
                        if mrow["low"] <= tp:
                            exit_price = tp
                            result_type = "TP"
                            close_ts = ts
                            break
                        elif mrow["high"] >= sl:
                            exit_price = sl
                            result_type = "SL"
                            close_ts = ts
                            break

                if exit_price is not None:
                    # Применяем проскальзывание
                    exit_price = self._apply_slippage(exit_price, "short" if position == "long" else "long", 
                                                    self.strategy.atr(df_cache_for_signal)[-1])
                    
                    price_change_pct = (exit_price - entry_price) / entry_price
                    if position == "short":
                        price_change_pct *= -1

                    pnl = qty_usd * price_change_pct
                    fee = self._apply_fee(qty_usd)
                    pnl -= fee
                    self.balance += pnl

                    self.equity_curve.append(self.balance)
                    try:
                        self.equity_timestamps.append(df_main.iloc[int(i)].name)
                    except:
                        self.equity_timestamps.append(df_main.index[i])

                    self._record_trade(
                        open_index, i, open_ts, close_ts, position, entry_price, exit_price,
                        tp, sl, qty_usd, qty_asset, balance_before, self.balance, result_type
                    )
                    position = None
                    open_index = None

            cur_day = df_main.iloc[i].name.date()

            if cur_day != start_of_day:
                start_of_day = cur_day
                day_start_balance = self.balance
                skip_day = False

            daily_drawdown = max(0.0, (day_start_balance - self.balance) / day_start_balance * 100)
            if daily_drawdown >= max_daily_dd:
                skip_day = True

            if skip_day:
                new_row = df_main.iloc[i:i + 1]
                df_cache = pd.concat([df_cache, new_row]).iloc[-warmup_bars:]
                continue

            if position is None and signal != 0:
                entry_price_raw = df_main.iloc[i]["open"]
                # Применяем проскальзывание при входе
                atr_vals = self.strategy.atr(df_cache_for_signal)
                atr_val = atr_vals[-1] if len(atr_vals) > 0 else entry_price_raw * 0.02
                entry_price = self._apply_slippage(entry_price_raw, "long" if signal > 0 else "short", atr_val)
                
                open_index = i
                position = "long" if signal > 0 else "short"

                # === ИДЕАЛЬНЫЙ TP/SL НА ОСНОВЕ СТРУКТУРЫ ===
                supports, resistances = self.strategy.find_key_levels(df_cache_for_signal, lookback=100)

                if position == "long":
                    tp = self.strategy.find_nearest_liquidity_zone(entry_price, resistances, "up")
                    sl = self.strategy.find_dynamic_sl(df_cache_for_signal, "long")
                else:
                    tp = self.strategy.find_nearest_liquidity_zone(entry_price, supports, "down")
                    sl = self.strategy.find_dynamic_sl(df_cache_for_signal, "short")

                # Защита от слишком узких уровней
                min_tp_dist = atr_val * 1.5
                actual_tp_dist = abs(tp - entry_price)
                if actual_tp_dist < min_tp_dist:
                    tp = entry_price + min_tp_dist if position == "long" else entry_price - min_tp_dist

                min_sl_dist = atr_val * 0.8
                actual_sl_dist = abs(entry_price - sl)
                if actual_sl_dist < min_sl_dist:
                    sl = entry_price - min_sl_dist if position == "long" else entry_price + min_sl_dist

                risk_amount = self.balance * getattr(self.cfg, "risk_percentage", 0.01)
                risk_per_unit = abs(entry_price - sl)
                leverage = getattr(self.cfg, "leverage", 1)
                if risk_per_unit > 0:
                    qty_usd = risk_amount * leverage * (entry_price / risk_per_unit)
                    qty_asset = qty_usd / entry_price
                else:
                    qty_usd = qty_asset = 0.0

                balance_before = self.balance

            new_row = df_main.iloc[i:i + 1]
            df_cache = pd.concat([df_cache, new_row]).iloc[-warmup_bars:]

        return self._generate_report()

    def _record_trade(self, open_index, close_index, open_ts, close_ts, side, entry_price, exit_price,
                      tp, sl, qty_usd, qty_asset, balance_before, balance_after, result_type):
        pnl = balance_after - balance_before
        pnl_pct = (pnl / balance_before * 100) if balance_before else 0.0
        trade = {
            "open_index": open_index,
            "open_ts": open_ts,
            "close_index": close_index,
            "close_ts": close_ts,
            "signal": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "tp": tp,
            "sl": sl,
            "qty_usd": qty_usd,
            "qty_asset": qty_asset,
            "result": result_type,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "balance_before": balance_before,
            "balance_after": balance_after,
        }
        self.trades.append(trade)

    def _generate_report(self):
        df_trades = pd.DataFrame(self.trades)

        if len(self.equity_curve) > 1:
            eq = pd.Series(self.equity_curve, index=self.equity_timestamps)
        else:
            eq = pd.Series([self.initial_balance], index=[None])

        hwm = eq.cummax()
        dd = (eq - hwm) / hwm
        max_dd = dd.min() if len(dd) else 0

        if len(eq) > 2:
            ret = np.log(eq / eq.shift(1)).dropna()
            sharpe = ret.mean() / ret.std() * np.sqrt(365) if ret.std() != 0 else 0
        else:
            sharpe = 0

        if len(df_trades) == 0:
            report = {
                "Total Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "Final Balance": self.balance,
                "Initial Balance": self.initial_balance,
                "Winrate %": 0.0,
                "Max Drawdown": float(max_dd),
                "Sharpe Ratio": float(sharpe)
            }
            logger.info("Нет сделок за период бэктеста")
        else:
            wins = len(df_trades[df_trades["result"] == "TP"])
            losses = len(df_trades[df_trades["result"] == "SL"])
            winrate = round(wins / len(df_trades) * 100, 2) if len(df_trades) > 0 else 0.0

            report = {
                "Total Trades": len(df_trades),
                "Wins": wins,
                "Losses": losses,
                "Final Balance": self.balance,
                "Initial Balance": self.initial_balance,
                "Winrate %": winrate,
                "Max Drawdown": float(max_dd),
                "Sharpe Ratio": float(sharpe)
            }

        trades_file = os.path.join(self.results_dir, f"trades.csv")
        df_trades.to_csv(trades_file, index=False)
        logger.info(f"Сохранено {len(df_trades)} трейдов")

        logger.info(f'Результат: total trades: {report["Total Trades"]}, '
                    f'winrate: {report["Winrate %"]}, '
                    f'final balance: {report["Final Balance"]:.2f},'
                    f'max_dd: {report["Max Drawdown"]:.2f},'
                    f'sharpe: {report["Sharpe Ratio"]:.2f}')

        try:
            plt.figure(figsize=(12, 5))
            plt.plot(eq.index, eq.values, label="Equity Curve", color="green")

            # --- МАРКЕРЫ СДЕЛОК ---
            if not df_trades.empty:
                for _, trade in df_trades.iterrows():
                    ts = trade['close_ts']
                    balance = trade['balance_after']
                    color = 'lime' if trade['result'] == 'TP' else 'red'
                    plt.scatter(ts, balance, color=color, s=30, zorder=5)

            plt.title("Equity Curve with Trade Markers")
            plt.xlabel("Time")
            plt.ylabel("Balance")
            plt.grid(True)
            plt.legend()

            plot_path = os.path.join(self.results_dir, "equity_curve.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"График сохранён: {plot_path}")

        except Exception as e:
            logger.exception(f"Ошибка сохранения графика: {e}")

        return report