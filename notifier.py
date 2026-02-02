# notifier.py
import logging
import requests
from typing import Optional, List

class TelegramNotifier:
    def __init__(self, bot_token: Optional[str], chat_ids: list[str]):
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.logger = logging.getLogger("telegram")

    def send_message(self, message: str) -> bool:
        if not self.bot_token or not self.chat_ids:
            self.logger.debug("Telegram not configured, skip message.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        success = True

        for chat_id in self.chat_ids:
            try:
                r = requests.post(
                    url,
                    json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                    timeout=10,
                )
                r.raise_for_status()
            except Exception:
                success = False
                self.logger.exception("Failed to send Telegram message to %s", chat_id)

        return success

    def send_daily_report(self, date: str, balance_change: float, winrate: float, total_trades: int, loss_reasons: List[str]):
        reasons_text = "\n".join(f"• {r}" for r in loss_reasons[:3]) if loss_reasons else "—"
        msg = (
            f"📊 <b>Ежедневный отчёт</b>\n"
            f"Дата: {date}\n"
            f"Изменение баланса: {balance_change:+.2f}%\n"
            f"Сделок: {total_trades}\n"
            f"Winrate: {winrate:.1f}%\n"
            f"Причины убытков:\n{reasons_text}"
        )
        self.send_message(msg)