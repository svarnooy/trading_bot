# trading_bot.py
import logging
import decimal
import time
import json
import os
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from .config import ConfigModel
from .notifier import TelegramNotifier
from .analytics import TradeAnalyzer
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("trading_bot")

# === ПОДМОДУЛИ ===
class SignalEngine:
    def __init__(self, cfg: ConfigModel, trading_bot):
        self.cfg = cfg
        self.bot = trading_bot  # ссылка на TradingBot вместо ExchangeClient

    def get_signal_and_levels(self, symbol: str, timeframe: str):
        """Возвращает сигнал и уровни TP/SL"""
        df = self.bot.fetch_klines(symbol, timeframe, limit=200)
        df_1h = self.bot.fetch_klines(symbol, "60", limit=100)
        
        if df is None or df.empty or df_1h is None or df_1h.empty:
            return None, None, None, None, None

        from .strategy_factory import create_strategy
        from .market_regime_detector import detect_market_regime

        current_regime = detect_market_regime(df_1h)
        now = datetime.now(timezone.utc)
        should_change = (
            self.bot._last_strategy_change is None or
            (now - self.bot._last_strategy_change).total_seconds() >= 4 * 3600 or
            self.bot._current_regime != current_regime
        )

        if should_change:
            # Логируем изменение режима
            from .regime_logger import RegimeLogger
            logger_obj = RegimeLogger()
            old_regime = self.bot._current_regime or "unknown"
            logger_obj.log_regime_change(symbol, old_regime, current_regime, now)
            
            strategy = create_strategy(self.cfg, df_1h)
            strategy.df_higher = self.bot.fetch_klines(symbol, "240", limit=100)
            self.bot._current_strategy = strategy
            self.bot._current_regime = current_regime
            self.bot._last_strategy_change = now
        else:
            strategy = self.bot._current_strategy

        signals = strategy.generate_signals(df)
        if signals is None or len(signals) < 3:
            return None, None, None, None, None

        signal = signals[-1]
        new_side = "Buy" if signal == 1 else "Sell" if signal == -1 else None

        # === ИДЕАЛЬНЫЙ TP/SL НА ОСНОВЕ СТРУКТУРЫ ===
        atr_val = strategy.atr(df)[-1]
        entry = df['close'].iloc[-1]
        supports, resistances = strategy.find_key_levels(df, lookback=100)

        if new_side == "Buy":
            tp = strategy.find_nearest_liquidity_zone(entry, resistances, "up")
            sl = strategy.find_dynamic_sl(df, "long")
        elif new_side == "Sell":
            tp = strategy.find_nearest_liquidity_zone(entry, supports, "down")
            sl = strategy.find_dynamic_sl(df, "short")
        else:
            tp = sl = None

        # Защита от слишком узких уровней
        if new_side is not None:
            min_tp_dist = atr_val * 1.5
            actual_tp_dist = abs(tp - entry)
            if actual_tp_dist < min_tp_dist:
                tp = entry + min_tp_dist if new_side == "Buy" else entry - min_tp_dist

            min_sl_dist = atr_val * 0.8
            actual_sl_dist = abs(entry - sl)
            if actual_sl_dist < min_sl_dist:
                sl = entry - min_sl_dist if new_side == "Buy" else entry + min_sl_dist

        return new_side, entry, tp, sl, strategy

class RiskController:
    def __init__(self, cfg: ConfigModel, trading_bot):
        self.cfg = cfg
        self.bot = trading_bot

    def compute_position_size(self, entry_price: float, stop_price: float, balance_usd: float) -> float:
        risk_amount = balance_usd * self.cfg.risk_percentage
        
        # Учёт комиссии: вход + выход
        fee_rate = getattr(self.cfg, "fee", 0.0011)
        total_fee_factor = 1 + (2 * fee_rate)
        
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit < 1e-8:
            raise ValueError("Stop distance too small")
        
        effective_risk = risk_amount / total_fee_factor
        qty_usd = effective_risk * self.cfg.leverage * (entry_price / risk_per_unit)
        return qty_usd

class OrderManager:
    def __init__(self, cfg: ConfigModel, trading_bot, notifier: TelegramNotifier):
        self.cfg = cfg
        self.bot = trading_bot
        self.notifier = notifier

    def close_partial(self, side, qty, reason="partial"):
        opposite = "Sell" if side == "Buy" else "Buy"
        res = self.bot.place_order(opposite, qty, reduce_only=True)
        logger.info(f"Частичное закрытие ({reason}): {qty}")
        return res

    def send_trade_notification(self, new_side, entry, tp, sl):
        self.notifier.send_message(
            f"🔔 <b>Новая позиция</b>\n"
            f"Монета: {self.cfg.symbol}\n"
            f"Направление: {'🟢 Лонг' if new_side == 'Buy' else '🔴 Шорт'}\n"
            f"Вход: ${entry:.2f}\n"
            f"TP: ${tp:.2f} | SL: ${sl:.2f}\n"
            f"Риск: {self.cfg.risk_percentage * 100:.1f}%"
        )

# === ОСНОВНОЙ БОТ ===
class TradingBot:
    def __init__(self, cfg: ConfigModel, portfolio_manager=None):
        from .notifier import TelegramNotifier
        self.cfg = cfg
        self.notifier = TelegramNotifier(bot_token=cfg.telegram_bot_token, chat_ids=cfg.telegram_chat_ids)
        self.portfolio_manager = portfolio_manager
        
        # Инициализация биржевого клиента
        self.session = None
        self._qty_step_cache = None
        self.df_cache = {}
        self._connect()
        logger.info("=== Этап 1: Сессия создана ===")
        
        # Подмодули
        self.signal_engine = SignalEngine(cfg, self)
        self.risk_controller = RiskController(cfg, self)
        self.order_manager = OrderManager(cfg, self, self.notifier)
        self.analyzer = TradeAnalyzer()
        
        # Состояние
        self.logger = logging.getLogger("TradingBot")
        self._last_health_check = datetime.now(timezone.utc)
        self._full_qty = 0.0
        self._entry_price = 0.0
        self._partial_closed = False
        self._network_alert_sent = False
        self._last_losses = []
        self._pause_until = None
        self._current_position_side = None
        self._last_strategy_change = None
        self._current_regime = None
        self._current_strategy = None
        
        self._load_state()
        logger.info("=== Этап 2: Состояние загружено ===")
        
        self._restore_position_on_startup()
        logger.info("=== Этап 3: Позиция восстановлена ===")
        
        self._send_startup_message()
        logger.info("=== Этап 4: Стартовое уведомление отправлено ===")
        
        self._run_optimization_if_needed()
        logger.info("=== Этап 5: Оптимизация завершена ===")
        
        logger.info("✅ БОТ ГОТОВ К ТОРГОВЛЕ")

    def _connect(self):
        try:
            from pybit.unified_trading import HTTP
            self.session = HTTP(
                testnet=self.cfg.testnet,
                demo=getattr(self.cfg, 'demo', False),
                api_key=self.cfg.api_key,
                api_secret=self.cfg.api_secret
            )
            logger.info("HTTP session created")
        except Exception as e:
            logger.warning("pybit not available or failed to init — live trading disabled. Error: %s", e)
            self.session = None

    def safe_request(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "network" in str(e).lower() or "timeout" in str(e).lower():
                    logger.warning(f"Network error (attempt {attempt+1}/3): {e}. Retrying in 5s...")
                    time.sleep(5)
                    self._connect()
                else:
                    raise
        return None

    def fetch_klines(self, symbol: str, interval: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        if not self.session:
            return None

        cache_key = (symbol, interval)
        df_cache = self.df_cache.get(cache_key)
        if df_cache is not None and not df_cache.empty:
            now = datetime.now(timezone.utc)
            last_ts = df_cache.index[-1]
            tf_minutes = float(self.cfg.timeframe)
            next_expected = last_ts + pd.Timedelta(minutes=tf_minutes)
            if now < next_expected:
                return df_cache.copy()

        if limit is None:
            limit = 200

        resp = self.safe_request(self.session.get_kline, symbol=symbol, interval=interval, limit=limit)
        if not resp or resp.get("retCode") != 0 or "list" not in resp.get("result", {}):
            return df_cache.copy() if df_cache is not None else None

        df = pd.DataFrame(resp["result"]["list"])
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        if df.shape[1] >= len(expected_cols):
            df = df.iloc[:, :len(expected_cols)]
            df.columns = expected_cols

        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        for c in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df.set_index('timestamp').sort_index()

        if len(df) > 0:
            df = df.iloc[:-1]

        cache_size = getattr(self.cfg, "cache_size", 50)
        if len(df) > cache_size:
            df = df.iloc[-cache_size:]

        self.df_cache[cache_key] = df.copy()
        return df[['open', 'high', 'low', 'close', 'volume', 'turnover']].copy()

    def get_wallet_balance(self) -> Optional[float]:
        if self.cfg.dry_run:
            return 1000.0
        if not self.session:
            return None
        resp = self.safe_request(self.session.get_wallet_balance, accountType='UNIFIED', coin='USDT')
        if resp and resp.get('retCode') == 0 and resp['result']['list']:
            return float(resp['result']['list'][0]['totalWalletBalance'])
        return None

    def get_position_status(self, symbol: Optional[str] = None) -> Tuple[str, float, float, float]:
        if self.cfg.dry_run or not self.session:
            return "None", 0.0, 0.0, 0.0
        try:
            symbol = symbol or self.cfg.symbol
            resp = self.safe_request(self.session.get_positions, category="linear", symbol=symbol)
            if resp and resp.get("retCode") == 0:
                positions = resp.get("result", {}).get("list", [])
                for p in positions:
                    size = float(p.get("size", 0.0))
                    if size > 0:
                        side = p.get("side", "None")
                        entry_price = float(p.get("avgPrice", 0.0))
                        pnl = float(p.get("unrealizedPnl", 0.0))
                        return side, size, entry_price, pnl
            return "None", 0.0, 0.0, 0.0
        except Exception:
            logger.exception("get_position_status error")
            return "None", 0.0, 0.0, 0.0

    def get_qty_step(self, symbol: str) -> float:
        if self._qty_step_cache:
            return self._qty_step_cache
        if not self.session:
            return 0.001
        resp = self.safe_request(self.session.get_instruments_info, category='linear', symbol=symbol)
        step = float(resp['result']['list'][0]['lotSizeFilter']['qtyStep'])
        self._qty_step_cache = step
        return step

    def precise_format_to_precision(self, number, precision):
        d = decimal.Decimal(str(number))
        dec_places = len(str(precision).split('.')[1]) if '.' in str(precision) else 0
        with decimal.localcontext() as ctx:
            ctx.rounding = decimal.ROUND_HALF_UP
            return float(round(d, dec_places))

    def place_order(self, side, qty, tp=None, sl=None, reduce_only=False):
        if self.cfg.dry_run or not self.session:
            logger.info("DRY RUN order: %s %s tp=%s sl=%s reduce=%s", side, qty, tp, sl, reduce_only)
            return {"dry_run": True}
        try:
            params = {
                "category": "linear",
                "symbol": self.cfg.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "GTC",
                "reduceOnly": reduce_only
            }
            if tp:
                params["takeProfit"] = str(tp)
            if sl:
                params["stopLoss"] = str(sl)
            return self.session.place_order(**params)
        except Exception:
            logger.exception("Order failed")
            return {"status": "error"}

    def _save_state(self):
        """Сохраняет текущее состояние в state.json"""
        state = {
            "symbol": self.cfg.symbol,
            "position": {
                "side": self._current_position_side,
                "size": self._full_qty,
                "entry_price": self._entry_price,
                "partial_closed": self._partial_closed
            },
            "last_losses": self._last_losses,
            "pause_until": self._pause_until.isoformat() if self._pause_until else None,
        }
        try:
            with open("state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("Состояние сохранено в state.json")
        except Exception as e:
            logger.exception("Ошибка сохранения состояния: %s", e)

    def _load_state(self):
        """Загружает состояние из state.json при старте"""
        if not os.path.exists("state.json"):
            return
        try:
            with open("state.json", "r") as f:
                state = json.load(f)
            if state.get("symbol") == self.cfg.symbol:
                pos = state.get("position", {})
                self._full_qty = float(pos.get("size", 0.0))
                self._entry_price = float(pos.get("entry_price", 0.0))
                self._partial_closed = bool(pos.get("partial_closed", False))
                self._current_position_side = pos.get("side")
                
                pause_str = state.get("pause_until")
                if pause_str:
                    self._pause_until = datetime.fromisoformat(pause_str.replace("Z", "+00:00"))
                
                self._last_losses = state.get("last_losses", [])
                logger.info("Состояние загружено из state.json")
        except Exception as e:
            logger.exception("Ошибка загрузки состояния: %s", e)

    def _restore_position_on_startup(self):
        """Восстанавливает позицию с биржи и обновляет локальное состояние"""
        try:
            # Добавляем таймаут для первого запроса
            start = time.time()
            pos_side, pos_size, entry_price, pnl = self.get_position_status(self.cfg.symbol)
            elapsed = time.time() - start
            logger.info(f"Запрос позиции выполнен за {elapsed:.2f} сек")
            
            if pos_size > 0:
                self._full_qty = pos_size
                self._entry_price = entry_price
                self._partial_closed = False
                self._current_position_side = pos_side
                logger.info("Восстановлена позиция с биржи: %s %.4f @ %.2f", pos_side, pos_size, entry_price)
                try:
                    self.notifier.send_message(f"✅ Восстановлена позиция: {pos_side} {pos_size:.4f}")
                except:
                    logger.warning("Не удалось отправить уведомление о позиции")
                
                # Регистрируем в портфельном менеджере
                if self.portfolio_manager:
                    self.portfolio_manager.register_position(self.cfg.symbol, pos_size, entry_price, pos_side)
            else:
                self._full_qty = 0.0
                self._entry_price = 0.0
                self._partial_closed = False
                self._current_position_side = None
                logger.info("Активных позиций не обнаружено")
        except Exception as e:
            logger.exception("Ошибка восстановления позиции: %s", e)
            logger.warning("Продолжаем без восстановления позиции")

    def _send_startup_message(self):
        try:
            msg = f"🚀 Trading bot started\nSymbol: {self.cfg.symbol}\nTimeframe: {self.cfg.timeframe}m"
            self.notifier.send_message(msg)
            logger.info("Стартовое уведомление отправлено в Telegram")
        except Exception as e:
            logger.warning("Не удалось отправить стартовое уведомление: %s", e)
            logger.info("Продолжаем без уведомления")

    def _notify_network_restored(self):
        """Уведомление о восстановлении интернета"""
        if self._network_alert_sent:
            self.notifier.send_message("✅ Интернет восстановлен. Торговля продолжается.")
            self._network_alert_sent = False

    def _notify_partial_close(self, qty, reason="50% TP"):
        """Уведомление о частичном закрытии"""
        self.notifier.send_message(
            f"CloseOperation 📉\n"
            f"Монета: {self.cfg.symbol}\n"
            f"Частичное закрытие: {qty:.4f} ({reason})"
        )

    def _notify_full_close(self, side, qty, pnl=0):
        """Уведомление о полном закрытии"""
        direction = "🟢 Лонг" if side == "Buy" else "🔴 Шорт"
        self.notifier.send_message(
            f"OperationContract 📊\n"
            f"Монета: {self.cfg.symbol}\n"
            f"Полное закрытие: {direction} {qty:.4f}\n"
            f"PNL: ${pnl:.2f}"
        )

    def _notify_pause_after_losses(self):
        """Уведомление о паузе после убытков"""
        self.notifier.send_message(
            "⏸️ Пауза на 6 часов\n"
            "Достигнуто 3 убытка подряд"
        )

    def _notify_optimization_complete(self, best_strategy):
        """Уведомление об окончании оптимизации"""
        self.notifier.send_message(
            f"🔧 Оптимизация завершена\n"
            f"Актив: {self.cfg.symbol}\n"
            f"Лучшая стратегия: {best_strategy}"
        )

    def _health_check(self):
        now = datetime.now(timezone.utc)
        if (now - self._last_health_check).total_seconds() >= 3 * 3600:
            try:
                balance = self.get_wallet_balance()
                pos_side, pos_size, entry, pnl = self.get_position_status(self.cfg.symbol)
                status = f"✅ {pos_side}" if pos_size > 0 else "❌ Нет"
                
                # Статистика за 7 дней
                stats = self.analyzer.get_stats(days=7)
                stats_msg = (
                    f"\n📊 <b>Статистика за 7 дней</b>\n"
                    f"Сделок: {stats['total_trades']}\n"
                    f"Winrate: {stats['winrate']}%\n"
                    f"Profit Factor: {stats['profit_factor']}"
                )
                
                self.notifier.send_message(
                    f"🛠️ <b>Бот онлайн</b>\nВремя: {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"Баланс: ${balance:.2f}\nПозиция: {status}{stats_msg}"
                )
                self._last_health_check = now
                logger.info("Health check выполнен успешно")
            except Exception:
                logger.exception("Health check failed")

    def _run_optimization_if_needed(self):
        """ВРЕМЕННО ОТКЛЮЧЕНО ДЛЯ ДЕМО-ТОРГОВЛИ"""
        logger.info("Оптимизация отключена для демо-торговли")
        # Отправляем финальное уведомление о готовности
        try:
            self.notifier.send_message(
                f"✅ <b>Бот готов к торговле</b>\n"
                f"Актив: {self.cfg.symbol}\n"
                f"Ожидание позиций..."
            )
        except:
            logger.warning("Не удалось отправить уведомление о готовности")

    def one_iteration(self):
        now = datetime.now(timezone.utc)
        
        # === ПАУЗА ПРИ 3 УБЫТКАХ ===
        if self._pause_until and now < self._pause_until:
            logger.info("Пауза до %s", self._pause_until)
            self._save_state()
            return {"status": "paused"}

        self._health_check()

        pos_side, pos_size, entry_price, pnl = self.get_position_status(self.cfg.symbol)
        current_position = pos_side if pos_size > 0 else None

        # Получаем сигнал и уровни
        new_side, entry, tp, sl, strategy = self.signal_engine.get_signal_and_levels(
            self.cfg.symbol, self.cfg.timeframe
        )

        if new_side is None:
            # Обновляем трейлинг даже без сигнала
            if current_position and pos_size > 0:
                self._update_trailing_sl(current_position, pos_size, strategy)
            self._save_state()
            return {"status": "monitoring"}

        # --- РЕВЕРС ПОЗИЦИИ ---
        if current_position and new_side and current_position != new_side:
            opposite = "Sell" if current_position == "Buy" else "Buy"
            close_result = self.place_order(opposite, pos_size, reduce_only=True)
            time.sleep(1)
            
            # УВЕДОМЛЕНИЕ О ПОЛНОМ ЗАКРЫТИИ
            if close_result.get("dry_run") or "retCode" in str(close_result):
                self._notify_full_close(current_position, pos_size, pnl)
            
            bal = self.get_wallet_balance() or 1000.0
            qty_usd = self.risk_controller.compute_position_size(entry, sl, bal)
            qty_asset = qty_usd / entry
            qty_step = self.get_qty_step(self.cfg.symbol)
            final_qty = self.precise_format_to_precision(qty_asset, qty_step)
            
            order_result = self.place_order(new_side, final_qty, tp=tp, sl=sl)
            if order_result.get("dry_run") or "retCode" in str(order_result):
                self.order_manager.send_trade_notification(new_side, entry, tp, sl)
                self.analyzer.save_trade(
                    self.cfg.symbol, new_side, entry, entry, tp, sl, 0, "ENTRY"
                )
                # Обновляем портфель
                if self.portfolio_manager:
                    self.portfolio_manager.close_position(self.cfg.symbol)
                    self.portfolio_manager.register_position(self.cfg.symbol, final_qty, entry, new_side)
            
            self._full_qty = final_qty
            self._entry_price = entry
            self._partial_closed = False
            self._current_position_side = new_side
            
            self._save_state()
            return {"status": "reversed"}

        # --- ОТКРЫТИЕ НОВОЙ ПОЗИЦИИ ---
        if not current_position and new_side:
            # Проверяем, можно ли открывать позицию
            if self.portfolio_manager and not self.portfolio_manager.can_open_position(self.cfg.symbol):
                logger.info("Открытие позиции отклонено portfolio manager")
                self._save_state()
                return {"status": "blocked"}
            
            bal = self.get_wallet_balance() or 1000.0
            qty_usd = self.risk_controller.compute_position_size(entry, sl, bal)
            qty_asset = qty_usd / entry
            qty_step = self.get_qty_step(self.cfg.symbol)
            final_qty = self.precise_format_to_precision(qty_asset, qty_step)
            
            order_result = self.place_order(new_side, final_qty, tp=tp, sl=sl)
            if order_result.get("dry_run") or "retCode" in str(order_result):
                self.order_manager.send_trade_notification(new_side, entry, tp, sl)
                self.analyzer.save_trade(
                    self.cfg.symbol, new_side, entry, entry, tp, sl, 0, "ENTRY"
                )
                # Регистрируем в портфельном менеджере
                if self.portfolio_manager:
                    self.portfolio_manager.register_position(self.cfg.symbol, final_qty, entry, new_side)
            
            self._full_qty = final_qty
            self._entry_price = entry
            self._partial_closed = False
            self._current_position_side = new_side
            
            self._save_state()
            return {"status": "opened"}

        # --- ЧАСТИЧНОЕ ЗАКРЫТИЕ + ТРЕЙЛИНГ ---
        if current_position and pos_size > 0:
            self._update_trailing_sl(current_position, pos_size, strategy)

        self._save_state()
        return {"status": "monitoring"}

    def _update_trailing_sl(self, current_position, pos_size, strategy):
        df = self.fetch_klines(self.cfg.symbol, self.cfg.timeframe, limit=200)
        if df is None or df.empty:
            return
            
        current_price = df['close'].iloc[-1]
        atr_val = strategy.atr(df)[-1]
        tp_dist = abs(self._entry_price - (self._entry_price + atr_val * 2.5))  # пример
        move = abs(current_price - self._entry_price)

        if move >= tp_dist * 0.5 and not self._partial_closed:
            partial_qty = pos_size * 0.5
            self.order_manager.close_partial(current_position, partial_qty, "50% TP")
            self._partial_closed = True
            # УВЕДОМЛЕНИЕ О ЧАСТИЧНОМ ЗАКРЫТИИ
            self._notify_partial_close(partial_qty, "50% TP")

        trail_dist = atr_val * 1.0
        if current_position == "Buy":
            new_sl = max(self._entry_price, current_price - trail_dist)
            if new_sl > self._entry_price:
                self.place_order("Buy", pos_size, sl=new_sl)
        else:
            new_sl = min(self._entry_price, current_price + trail_dist)
            if new_sl < self._entry_price:
                self.place_order("Sell", pos_size, sl=new_sl)

    def run(self):
        import time
        from datetime import datetime, timezone, timedelta

        last_network_ok = True
        start_balance = self.get_wallet_balance()
        max_daily_dd = getattr(self.cfg, "max_daily_drawdown_pct", 2.0)
        last_daily_check = datetime.now(timezone.utc).date()

        while True:
            try:
                now = datetime.now(timezone.utc)
                cur_balance = self.get_wallet_balance()
                if cur_balance is not None and start_balance:
                    daily_dd = max(0.0, (start_balance - cur_balance) / start_balance * 100)
                    if daily_dd >= max_daily_dd:
                        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                        wake_time = tomorrow + timedelta(minutes=5)
                        sleep_sec = (wake_time - now).total_seconds()
                        
                        # УВЕДОМЛЕНИЕ О ДОСТИЖЕНИИ ЛИМИТА ПРОСАДКИ
                        self.notifier.send_message(
                            f"⚠️ <b>ДОСТИГНУТ ЛИМИТ ПРОСАДКИ</b>\n"
                            f"Текущая просадка: {daily_dd:.2f}%\n"
                            f"Максимум: {max_daily_dd:.2f}%\n"
                            f"Бот засыпает до 00:05 UTC"
                        )
                        
                        logger.warning("Дневная просадка %.2f%% >= %.2f%% — сон до 00:05", daily_dd, max_daily_dd)
                        time.sleep(max(sleep_sec, 0))
                        continue

                if now.date() != last_daily_check:
                    last_daily_check = now.date()
                    start_balance = cur_balance

                res = self.one_iteration()
                logger.info("Iteration result: %s", res)
                
                # Проверка восстановления сети после успешной итерации
                if not last_network_ok:
                    self._notify_network_restored()
                    last_network_ok = True

            except Exception as e:
                logger.exception("Error in iteration: %s", e)
                if "network" in str(e).lower() or "timeout" in str(e).lower():
                    if last_network_ok:
                        self.notifier.send_message("⚠️ Проблема с интернетом. Бот работает в фоне.")
                        last_network_ok = False
                        self._network_alert_sent = True
                else:
                    self.notifier.send_message(f"❌ Критическая ошибка: {str(e)[:100]}")
                    break

            # === ИСПРАВЛЕННЫЙ БЛОК РАСЧЁТА ВРЕМЕНИ ===
            tf_minutes = int(self.cfg.timeframe)
            current_minute = now.minute
            next_interval = ((current_minute // tf_minutes) + 1) * tf_minutes

            if next_interval < 60:
                next_time = now.replace(minute=next_interval, second=2, microsecond=0)
            else:
                hours_to_add = next_interval // 60
                next_minute = next_interval % 60
                next_hour = (now.hour + hours_to_add) % 24
                next_time = now.replace(hour=next_hour, minute=next_minute, second=2, microsecond=0)

            if next_time <= now:
                next_time = next_time + pd.Timedelta(hours=1)

            sleep_sec = (next_time - now).total_seconds()
            time.sleep(max(sleep_sec, 1))