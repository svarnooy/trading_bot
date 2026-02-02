# bot/regime_logger.py
import json
import os
from datetime import datetime, timezone

class RegimeLogger:
    def __init__(self, log_file="regime_history.json"):
        self.log_file = log_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log_regime_change(self, symbol: str, old_regime: str, new_regime: str, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "old_regime": old_regime,
            "new_regime": new_regime
        }
        
        # Загружаем текущую историю
        with open(self.log_file, "r") as f:
            history = json.load(f)
        
        # Добавляем новую запись
        history.append(entry)
        
        # Сохраняем (оставляем последние 1000 записей)
        history = history[-1000:]
        
        with open(self.log_file, "w") as f:
            json.dump(history, f, indent=2)
    
    def get_recent_changes(self, symbol: str, limit: int = 10):
        with open(self.log_file, "r") as f:
            history = json.load(f)
        
        symbol_changes = [e for e in history if e["symbol"] == symbol]
        return symbol_changes[-limit:]