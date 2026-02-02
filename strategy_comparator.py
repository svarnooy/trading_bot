# bot/strategy_comparator.py
import os
import logging
import pandas as pd
from datetime import datetime, timezone
from .backtester import FastBacktester
from .strategy_factory import create_strategy

logger = logging.getLogger("StrategyComparator")

class StrategyComparator:
    def __init__(self, cfg, symbol: str, timeframe: str):
        self.cfg = cfg
        self.symbol = symbol
        self.timeframe = timeframe
        self.results_dir = f"strategy_comparison/{symbol}_{timeframe}"
        os.makedirs(self.results_dir, exist_ok=True)

    def compare_strategies(self, df_main, df_1h, df_240, df_m1, days_back=30):
        """
        Тестирует все 4 стратегии и возвращает лучшую
        """
        logger.info(f"Сравнение стратегий для {self.symbol} {self.timeframe}m")
        
        # Сокращаем данные до нужного периода
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days_back)
        df_main = df_main[df_main.index >= cutoff]
        df_1h = df_1h[df_1h.index >= cutoff]
        df_240 = df_240[df_240.index >= cutoff]
        df_m1 = df_m1[df_m1.index >= cutoff]

        strategies = ["trend", "range", "breakout", "countertrend"]
        results = {}

        for strategy_name in strategies:
            try:
                # Создаём конфиг с фиксированной стратегией
                test_cfg = type(self.cfg)(**self.cfg.__dict__)
                test_cfg.forced_strategy = strategy_name  # добавим это в strategy_factory
                
                # Создаём стратегию
                strategy = create_strategy(test_cfg, df_1h)
                strategy.df_higher = df_240

                # Запускаем бэктест
                bt = FastBacktester(test_cfg, strategy, initial_balance=1000.0)
                report = bt.run(df_main, df_m1)

                if report["Total Trades"] >= 5:
                    results[strategy_name] = {
                        "final_balance": report["Final Balance"],
                        "winrate": report["Winrate %"],
                        "total_trades": report["Total Trades"],
                        "max_drawdown": report["Max Drawdown"],
                        "sharpe": report["Sharpe Ratio"]
                    }
                    logger.info(f"{strategy_name}: {report['Final Balance']:.2f} ({report['Winrate %']}%)")
                else:
                    logger.warning(f"{strategy_name}: недостаточно сделок ({report['Total Trades']})")

            except Exception as e:
                logger.exception(f"Ошибка при тестировании {strategy_name}: {e}")
                continue

        # Сохраняем результаты
        if results:
            results_df = pd.DataFrame(results).T
            results_df = results_df.sort_values("final_balance", ascending=False)
            results_df.to_csv(f"{self.results_dir}/strategy_comparison.csv", index=True)
            
            # Выбираем лучшую стратегию
            best_strategy = results_df.index[0]
            logger.info(f"Лучшая стратегия: {best_strategy}")
            return best_strategy
        
        logger.warning("Не найдено подходящих стратегий")
        return "trend"  # по умолчанию