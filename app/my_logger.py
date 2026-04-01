import logging
import os
from logging.handlers import RotatingFileHandler

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================

# Создаем директории для логов
os.makedirs("logs", exist_ok=True)

# Настройка основного логгера
logger = logging.getLogger('ml_predictor')
logger.setLevel(logging.INFO)

# Формат для текстовых логов
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Файловый handler с ротацией (максимум 10 файлов по 10MB)
file_handler = RotatingFileHandler(
    'logs/predictions.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=10,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)