# predict.py
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
from fer import FER

model = joblib.load("tpot_emotion_model.pkl")  # путь к модели
detector = FER(mtcnn=True)
FRAME_LIMIT = 20

if len(sys.argv) != 2:
    print("❗ Укажи путь: python predict.py /path/to/folder")
    sys.exit(1)

target_path = Path(sys.argv[1])
assert target_path.exists(), f"Путь {target_path} не существует"

results = []
for case in tqdm(sorted(target_path.iterdir()), desc="🔍 Анализ"):
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
                emotion_list.append(res[0]["emotions"])
        except Exception:
            continue

    if not emotion_list:
        continue

    arr = np.array([[e[k] for k in e] for e in emotion_list])
    keys = list(emotion_list[0].keys())
    features = {f"mean_{k}": np.mean(arr[:, i]) for i, k in enumerate(keys)}
    features.update({f"var_{k}": np.var(arr[:, i]) for i, k in enumerate(keys)})
    features["folder"] = str(case)
    results.append(features)

df = pd.DataFrame(results)
X = df.drop(columns=["folder"])
df["prediction"] = model.predict(X)

print("\n📊 Предсказания:")
print(df[["folder", "prediction"]])
