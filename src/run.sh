#!/bin/bash

# Проверка: путь к train передан
if [ -z "$1" ]; then
  echo "❌ Укажите путь к директории с train (например: ./run.sh /home/ubuntu/antifrod_biometry_dataset_publish/train)"
  exit 1
fi

TRAIN_PATH="$1"

# Опционально: активируем venv, если есть
if [ -f ".venv311/bin/activate" ]; then
  source .venv311/bin/activate
fi

# Установка переменной окружения и запуск
DATASET_ROOT="$TRAIN_PATH" python src/start.py
