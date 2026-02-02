# config.py
from pathlib import Path
import yaml
from pydantic import BaseModel
from typing import Optional, List, Union
import logging

logger = logging.getLogger("Config")

class ConfigModel(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_ids: Optional[List[Union[str, int]]] = None

    # Мульти-актив (основной список)
    symbols: Optional[List[dict]] = None
    
    # Для совместимости (если одна монета)
    symbol: str = "ETHUSDT"
    timeframe: str = "30"

    # Trading settings
    leverage: int = 3
    fee: float = 0.0011
    risk_percentage: float = 0.008
    atr_period: int = 14
    take_profit_atr: float = 3.0
    stop_loss_atr: float = 1.2
    short_sma_period: int = 21
    long_sma_period: int = 55
    cache_size: int = 200
    warmup_bars: int = 200
    max_daily_drawdown_pct: float = 2.0

    # Modes
    dry_run: bool = True
    testnet: bool = False
    demo: bool = True

    # Logging
    log_level: int = logging.INFO

    model_config = {
        "extra": "allow"
    }

def load_config(path: str = "config.yaml") -> ConfigModel:
    p = Path(path)
    cfg_dict = {}
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", path)
        except Exception as e:
            logger.exception("Failed to load configuration from %s: %s", path, e)
    else:
        logger.warning("Config file %s not found, using defaults", path)

    # Handle telegram block
    if "telegram" in cfg_dict:
        telegram_cfg = cfg_dict.pop("telegram") or {}
        cfg_dict["telegram_bot_token"] = telegram_cfg.get("bot_token")
        cfg_dict["telegram_chat_ids"] = telegram_cfg.get("chat_ids")

    # Fix log_level if it's a string
    if isinstance(cfg_dict.get("log_level"), str):
        level_str = cfg_dict["log_level"].upper()
        cfg_dict["log_level"] = getattr(logging, level_str, logging.INFO)

    return ConfigModel(**cfg_dict)

def setup_logging(level: int = logging.INFO):
    fmt = "%(asctime)s | %(levelname)5s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logger.info("Logging initialized at level %s", logging.getLevelName(level))