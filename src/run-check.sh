#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Укажи путь к папке с изображениями"
  echo "Пример: ./run.sh /home/ubuntu/antifrod_biometry_dataset_publish/to_check"
  exit 1
fi

echo "🚀 Запуск предсказания для папки: $1"
python predict.py "$1"
