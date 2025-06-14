# 📦 Импорты
import os
import cv2
import numpy as np
import pandas as pd
from fer import FER
from pathlib import Path
from tqdm import tqdm
import joblib
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==== Конфигурация ====
BASE_DIR = Path(__file__).resolve().parent
dataset_base = Path(os.environ.get("DATASET_ROOT", BASE_DIR))
DATASET_ROOTS = [dataset_base / "0", dataset_base / "1"]
LABEL_MAP = {"0": "ok", "1": "suspicious"}
FRAME_LIMIT = 20

# ==== Инициализация ====
print("🚀 Извлечение признаков из изображений...")
detector = FER(mtcnn=True)
results = []

# ==== Проход по папкам ====
for dataset_path in DATASET_ROOTS:
    label_name = dataset_path.name
    true_label = LABEL_MAP[label_name]

    for case in tqdm(sorted(dataset_path.iterdir()), desc=f"📂 Папка {label_name}"):
        if not case.is_dir():
            continue
        emotion_list = []
        for img_path in sorted(case.glob("*.jpg"))[:FRAME_LIMIT]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            try:
                res = detector.detect_emotions(img)
                if res:
                    emotion_list.append(res[0]['emotions'])
            except Exception:
                continue

        if not emotion_list:
            continue

        arr = np.array([[e[k] for k in e] for e in emotion_list])
        keys = list(emotion_list[0].keys())

        features = {f"mean_{k}": float(np.mean(arr[:, i])) for i, k in enumerate(keys)}
        features.update({f"var_{k}": float(np.var(arr[:, i])) for i, k in enumerate(keys)})
        features["label"] = true_label
        features["folder"] = str(case)
        results.append(features)

# ==== Сохранение признаков ====
df = pd.DataFrame(results)
df.to_csv(BASE_DIR / "emotion_features.csv", index=False)
print("✅ Признаки сохранены: emotion_features.csv")

# ==== Обучение модели TPOT ====
print("🧠 Обучение AutoML модели с TPOT...")
X = df.drop(columns=["label", "folder"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# Используем легкий конфиг, без QuantileTransformer
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    random_state=42,
    n_jobs=-1,
    config_dict='TPOT light'  # ✅ безопасный режим
)
tpot.fit(X_train, y_train)

# ==== Оценка ====
y_pred = tpot.predict(X_test)
print("\n📈 Классификационный отчёт:")
print(classification_report(y_test, y_pred))

# ==== Сохранение модели и кода ====
joblib.dump(tpot.fitted_pipeline_, BASE_DIR / "tpot_emotion_model.pkl")
tpot.export(BASE_DIR / "tpot_best_pipeline.py")

print("\n🎉 Модель обучена и сохранена:")
print(" - tpot_emotion_model.pkl")
print(" - tpot_best_pipeline.py")
