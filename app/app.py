import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import librosa
import numpy as np


MODEL_DIR = "models/"

# Загружаем модели и метки
available_models = [f.stem for f in Path(MODEL_DIR).glob("*.pkl") if f.stem != "label_encoder" and f.stem != "scaler"]
models_dict = {name: joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl")) for name in available_models}

le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

st.title("Sound Classification Dashboard")

# выбор моделей
selected_models = st.multiselect("Выберите модели для классификации", available_models, default=available_models)

# загрузка файла
uploaded_file = st.file_uploader("Загрузите WAV файл", type=["wav"])


def return_features_from_sound(dir_sound_file, local_sample_rate):

    signal, sr = librosa.load(dir_sound_file, sr = local_sample_rate) # загружаем файл


    # 3.1 Спектральный центроид
    centroid = librosa.feature.spectral_centroid(y=signal, sr=local_sample_rate)
    mean_centroid = np.mean(centroid)

    # 3.2 Спектральная ширина (bandwidth)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=local_sample_rate)
    mean_bandwidth = np.mean(bandwidth)

    # 3.3 Спектральная плоскость (flatness)
    flatness = librosa.feature.spectral_flatness(y=signal)
    mean_flatness = np.mean(flatness)

    # 3.4 Спектральный наклон (roll-off)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=local_sample_rate, roll_percent=0.85)
    mean_rolloff = np.mean(rolloff)

    # 3.5 MFCC (мел-частотные кепстральные коэффициенты) — полезны для классификации
    mfccs = librosa.feature.mfcc(y=signal, sr=local_sample_rate, n_mfcc=10)
    mean_mfccs = np.mean(mfccs, axis=1)

    # 3.6 RMS (корневая среднеквадратичная энергия)
    rms = librosa.feature.rms(y=signal)
    mean_rms = np.mean(rms)

    features_for_current_sound = [mean_centroid, mean_bandwidth, mean_flatness, mean_rolloff, mean_rms]
    for i in mean_mfccs:
        features_for_current_sound.append(i)
    features_for_current_sound_1 = list(map(lambda x: float(x), features_for_current_sound))
    return features_for_current_sound_1


if uploaded_file and selected_models:
    st.audio(uploaded_file, format="audio/wav")

    # сохраняем временно на диск
    tmp_path = f"temp_sound.wav"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # извлекаем признаки
    features = return_features_from_sound(tmp_path, 16000)  # freq_sounds = 16kHz
    X_new = np.array(features).reshape(1, -1)

    # масштабирование для моделей, которые требуют StandardScaler
    X_new_scaled = scaler.transform(X_new)

    st.write("### Результаты классификации:")

    for model_name in selected_models:
        model = models_dict[model_name]
        # выбираем нужный формат признаков
        if model_name in ["Logistic Regression", "KNN", "MLP", "SVM"]:
            probs = model.predict_proba(X_new_scaled)
        else:
            probs = model.predict_proba(X_new)

        top_idx = np.argmax(probs)
        pred_label = le.inverse_transform([top_idx])[0]
        st.write(f"**{model_name}:**")
        st.write(f"Предсказанный класс: {pred_label}")
        st.write(f"Вероятности по классам:")
        df_probs = pd.DataFrame(probs, columns=le.classes_)
        st.dataframe(df_probs)
