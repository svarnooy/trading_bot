# bot/analytics.py
import pandas as pd
import os
from datetime import datetime, timezone

class TradeAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_trade(self, symbol, side, entry, exit_price, tp, sl, pnl, result_type):
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "tp": tp,
            "sl": sl,
            "pnl": pnl,
            "result": result_type
        }
        
        file_path = os.path.join(self.results_dir, "trade_history.csv")
        df = pd.DataFrame([trade])
        
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

    def get_stats(self, days=7):
        file_path = os.path.join(self.results_dir, "trade_history.csv")
        if not os.path.exists(file_path):
            return {"total_trades": 0, "winrate": 0, "profit_factor": 0}
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]
        
        if len(df) == 0:
            return {"total_trades": 0, "winrate": 0, "profit_factor": 0}
        
        wins = df[df['result'] == 'TP']
        losses = df[df['result'] == 'SL']
        
        winrate = len(wins) / len(df) * 100
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "total_trades": len(df),
            "winrate": round(winrate, 1),
            "profit_factor": round(profit_factor, 2)
        }