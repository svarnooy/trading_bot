# bot/optimizer.py
import os
import logging
import pandas as pd
from itertools import product
from datetime import datetime, timezone
from .backtester import FastBacktester
from .strategy_factory import create_strategy

logger = logging.getLogger("Optimizer")

class ParameterOptimizer:
    def __init__(self, cfg, symbol: str, timeframe: str):
        self.cfg = cfg
        self.symbol = symbol
        self.timeframe = timeframe
        self.results_dir = f"optimization/{symbol}_{timeframe}"
        os.makedirs(self.results_dir, exist_ok=True)

    def optimize(self, df_main, df_1h, df_240, df_m1, days_back=30):
        """
        Оптимизирует параметры на последних N днях
        """
        logger.info(f"Запуск оптимизации для {self.symbol} {self.timeframe}m")
        
        # Сокращаем данные до нужного периода
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days_back)
        df_main = df_main[df_main.index >= cutoff]
        df_1h = df_1h[df_1h.index >= cutoff]
        df_240 = df_240[df_240.index >= cutoff]
        df_m1 = df_m1[df_m1.index >= cutoff]

        # Диапазоны параметров
        tp_ranges = [2.0, 2.5, 3.0, 3.5]
        sl_ranges = [1.0, 1.2, 1.5]
        atr_periods = [10, 14, 20]
        
        best_result = None
        best_params = None
        all_results = []

        for tp_mult, sl_mult, atr_period in product(tp_ranges, sl_ranges, atr_periods):
            # Пропускаем плохие RR
            if tp_mult / sl_mult < 1.5:
                continue
                
            # Создаём конфиг для теста
            test_cfg = type(self.cfg)(**self.cfg.__dict__)
            test_cfg.take_profit_atr = tp_mult
            test_cfg.stop_loss_atr = sl_mult
            test_cfg.atr_period = atr_period

            try:
                # Создаём стратегию
                strategy = create_strategy(test_cfg, df_1h)
                strategy.df_higher = df_240

                # Запускаем бэктест
                bt = FastBacktester(test_cfg, strategy, initial_balance=1000.0)
                report = bt.run(df_main, df_m1)

                if report["Total Trades"] < 5:
                    continue

                result = {
                    "tp_mult": tp_mult,
                    "sl_mult": sl_mult,
                    "atr_period": atr_period,
                    "final_balance": report["Final Balance"],
                    "winrate": report["Winrate %"],
                    "total_trades": report["Total Trades"],
                    "max_drawdown": report["Max Drawdown"]
                }
                all_results.append(result)

                # Сохраняем лучший результат
                if best_result is None or report["Final Balance"] > best_result["final_balance"]:
                    if report["Max Drawdown"] < 25.0:  # фильтр по просадке
                        best_result = result
                        best_params = {
                            "take_profit_atr": tp_mult,
                            "stop_loss_atr": sl_mult,
                            "atr_period": atr_period
                        }

            except Exception as e:
                logger.warning(f"Ошибка при тестировании {tp_mult}/{sl_mult}/{atr_period}: {e}")
                continue

        # Сохраняем все результаты
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values("final_balance", ascending=False)
            results_df.to_csv(f"{self.results_dir}/optimization_results.csv", index=False)
            
            # Сохраняем лучшие параметры
            if best_params:
                optimized_path = f"{self.results_dir}/optimized_params.yaml"
                import yaml
                with open(optimized_path, "w") as f:
                    yaml.dump(best_params, f, allow_unicode=True, sort_keys=False)
                logger.info(f"Лучшие параметры сохранены: {optimized_path}")
                return best_params
        
        logger.warning("Не найдено подходящих параметров")
        return None