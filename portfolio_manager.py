# bot/portfolio_manager.py
import logging
import threading
from datetime import datetime, timezone
from .notifier import TelegramNotifier

logger = logging.getLogger("PortfolioManager")

class PortfolioManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.notifier = TelegramNotifier(
            bot_token=cfg.telegram_bot_token, 
            chat_ids=cfg.telegram_chat_ids
        )
        self.positions = {}
        self.total_balance = 1000.0
        self.max_total_drawdown = getattr(cfg, "max_total_drawdown_pct", 5.0)
        self.lock = threading.Lock()

    def register_position(self, symbol: str, size: float, entry_price: float, side: str):
        """Регистрирует новую позицию"""
        with self.lock:
            self.positions[symbol] = {
                "size": size,
                "entry_price": entry_price,
                "side": side,
                "timestamp": datetime.now(timezone.utc)
            }
            logger.info(f"Позиция зарегистрирована: {symbol} {side} {size}")

    def close_position(self, symbol: str):
        """Закрывает позицию"""
        with self.lock:
            if symbol in self.positions:
                del self.positions[symbol]
                logger.info(f"Позиция закрыта: {symbol}")

    def update_balance(self, new_balance: float):
        """Обновляет общий баланс"""
        with self.lock:
            old_balance = self.total_balance
            self.total_balance = new_balance
            
            # Проверка общей просадки
            if old_balance > 0:
                drawdown = (old_balance - new_balance) / old_balance * 100
                if drawdown >= self.max_total_drawdown:
                    self.notifier.send_message(
                        f"⚠️ <b>ДОСТИГНУТ ЛИМИТ ОБЩЕЙ ПРОСАДКИ</b>\n"
                        f"Текущая просадка: {drawdown:.2f}%\n"
                        f"Максимум: {self.max_total_drawdown:.2f}%"
                    )

    def get_total_risk(self) -> float:
        """Возвращает общий риск по всем позициям"""
        with self.lock:
            total_risk = 0.0
            for pos in self.positions.values():
                # Здесь можно добавить расчёт риска по каждой позиции
                total_risk += self.cfg.risk_percentage
            return total_risk

    def can_open_position(self, symbol: str) -> bool:
        """Проверяет, можно ли открывать новую позицию"""
        with self.lock:
            # Максимум 5 одновременных позиций
            if len(self.positions) >= 5:
                return False
            
            # Общий риск не должен превышать 3%
            if self.get_total_risk() >= 0.03:
                return False
                
            return True

    def update_portfolio_status(self):
        """Обновляет статус портфеля и отправляет отчёт"""
        with self.lock:
            if self.positions:
                status_msg = "📊 <b>Портфель</b>\n"
                for symbol, pos in self.positions.items():
                    status_msg += f"{symbol}: {pos['side']} {pos['size']:.4f}\n"
                # Отправляем раз в час
                current_hour = datetime.now(timezone.utc).hour
                if not hasattr(self, '_last_report_hour') or self._last_report_hour != current_hour:
                    self.notifier.send_message(status_msg)
                    self._last_report_hour = current_hour