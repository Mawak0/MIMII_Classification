import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import librosa
from scipy.special import softmax


MODEL_DIR = "models/"

# =======================
#   Маппинг классов
# =======================

MAIN_CLASS_MAP = {
    0: "pump",
    1: "slider",
    2: "fan",
    3: "valve"
}

SUBCLASS_MAP = {
    0: "id_00",
    1: "id_02",
    2: "id_04",
    3: "id_06"
}

def decode_class(encoded_value, subclass_multiplier=100):
    main_class = encoded_value // subclass_multiplier
    sub_class = encoded_value % subclass_multiplier

    main_name = MAIN_CLASS_MAP.get(main_class, "unknown")
    sub_name = SUBCLASS_MAP.get(sub_class, f"id_{sub_class:02d}")

    return f"{main_name}_{sub_name}"


# =======================
#   Загрузка моделей
# =======================

available_models = [
    f.stem for f in Path(MODEL_DIR).glob("*.pkl")
    if f.stem not in ("label_encoder", "scaler")
]

models_dict = {
    name: joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
    for name in available_models
}

le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


# =======================
#   Извлечение признаков
# =======================

def return_features_from_sound(dir_sound_file, local_sample_rate):
    signal, sr = librosa.load(dir_sound_file, sr=local_sample_rate)

    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=local_sample_rate))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=local_sample_rate))
    flatness = np.mean(librosa.feature.spectral_flatness(y=signal))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=local_sample_rate, roll_percent=0.85))
    rms = np.mean(librosa.feature.rms(y=signal))
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=local_sample_rate, n_mfcc=10), axis=1)

    features = [centroid, bandwidth, flatness, rolloff, rms]
    features.extend(mfccs)
    return [float(x) for x in features]


# =======================
#   UI
# =======================

st.title("Sound Classification Dashboard")

selected_models = st.multiselect(
    "Выберите модели для классификации",
    available_models,
    default=available_models
)

uploaded_file = st.file_uploader("Загрузите WAV файл", type=["wav"])

if uploaded_file and selected_models:

    st.audio(uploaded_file, format="audio/wav")

    # сохраняем временно
    tmp_path = "temp_sound.wav"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # извлечение признаков
    features = return_features_from_sound(tmp_path, 16000)
    X_new = np.array(features).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)

    st.write("### Результаты классификации")

    # таблица вероятностей всех моделей
    results = []

    for model_name in selected_models:
        model = models_dict[model_name]

        # predict_proba или softmax
        if hasattr(model, "predict_proba"):
            # scaled только для тех, кто обучался на scaled
            if model_name in ["Logistic Regression", "KNN", "MLP", "SVM"]:
                probs = model.predict_proba(X_new_scaled)
            else:
                probs = model.predict_proba(X_new)
        else:
            # SVC без probability=True
            decision = model.decision_function(
                X_new_scaled if model_name in ["SVM"] else X_new
            )
            probs = softmax(decision, axis=1)

        probs = probs.flatten()
        top_idx = np.argmax(probs)
        raw_value = le.inverse_transform([top_idx])[0]
        pretty_label = decode_class(raw_value)

        # вывод под моделью
        st.subheader(f"🔹 {model_name}")
        st.write(f"Предсказано: **{pretty_label}**")

        # формируем таблицу
        decoded_labels = [decode_class(x) for x in le.classes_]
        row = {"model": model_name}
        for label, prob in zip(decoded_labels, probs):
            row[label] = float(prob)
        results.append(row)

    st.subheader("📊 Все вероятности")

    df_probs = pd.DataFrame(results)

    # Определяем числовые столбцы
    numeric_cols = df_probs.select_dtypes(include=['float', 'int']).columns

    # Красиво выводим
    st.dataframe(df_probs.style.format({col: "{:.3f}" for col in numeric_cols}))

