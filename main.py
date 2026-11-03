# -*- coding: utf-8 -*-
"""
Streamlit-приложение для классификации рукописных цифр (MNIST).
Адаптировано для Python 3.12 / Visual Studio Code.

Запуск:
    pip install -r requirements.txt
    streamlit run main.py
"""

import platform
import subprocess

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# ============================================================
# Блок 0 – Системная информация
# ============================================================
st.sidebar.header("🖥️ Системная информация")
st.sidebar.write(f"**Хост:** {platform.node()}")
st.sidebar.write(f"**Процессор:** {platform.processor()}")
st.sidebar.write(f"**Платформа:** {platform.platform()}")
st.sidebar.write(f"**Python:** {platform.python_version()}")
st.sidebar.write(f"**TensorFlow:** {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    st.sidebar.success(f"✅ GPU доступен: {gpus[0].name}")
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = details.get("device_name", "Неизвестно")
        st.sidebar.info(f"**Модель GPU:** {gpu_name}")
    except Exception:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                st.sidebar.info(f"**Модель GPU:** {result.stdout.strip()}")
            else:
                st.sidebar.info("Не удалось определить модель GPU")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            st.sidebar.info("nvidia-smi не найден в PATH")
else:
    st.sidebar.warning("⚠️ GPU не найден, обучение пойдёт на CPU.")
    st.sidebar.info(
        "💡 Для ускорения установите GPU-версию TensorFlow "
        "или запускайте на машине с CUDA."
    )

# Заголовок приложения
st.title("🧠 Классификация рукописных цифр (MNIST)")
st.markdown(
    "Обучите свою модель на датасете MNIST или используйте предобученную, "
    "затем загрузите изображение цифры для распознавания."
)


# ============================================================
# Блок 1 – Загрузка данных MNIST
# ============================================================
@st.cache_resource
def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()
st.success("Датасет MNIST загружен!")


def build_model() -> keras.Model:
    """Создаёт CNN с BatchNormalization и Dropout (используется в блоках 3 и 4)."""
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


# ============================================================
# Блок 2 – Боковая панель: параметры обучения
# ============================================================
st.sidebar.header("Настройки модели")
epochs = st.sidebar.slider("Количество эпох", min_value=1, max_value=15, value=5, step=1)
batch_size = st.sidebar.selectbox("Размер батча", [32, 64, 128, 256], index=1)
optimizer = st.sidebar.selectbox("Оптимизатор", ["adam", "sgd", "rmsprop"], index=0)


# ============================================================
# Блок 3 – Обучение по запросу пользователя
# ============================================================
if st.sidebar.button("🚀 Обучить модель"):
    with st.spinner("Обучение модели... Это может занять несколько минут."):
        model = build_model()
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=0,
        )
        st.session_state["model"] = model
        st.session_state["history"] = history
        st.sidebar.success("Модель обучена!")
        st.sidebar.text(
            f"Финальная точность на валидации: "
            f"{history.history['val_accuracy'][-1]:.4f}"
        )


# ============================================================
# Блок 4 – Базовая модель при первом запуске
# ============================================================
if "model" not in st.session_state:
    st.info(
        "Модель ещё не обучена. Создаём базовую модель "
        "на всех 60000 изображениях (5 эпох)..."
    )
    with st.spinner("Создание улучшенной базовой модели... (5 эпох)"):
        default_model = build_model()
        default_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        default_history = default_model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=5,
            validation_split=0.1,
            verbose=0,
        )
        st.session_state["model"] = default_model
        st.session_state["history"] = default_history
        st.success("✅ Улучшенная базовая модель готова!")
        st.info(
            f"Точность на валидации: "
            f"{default_history.history['val_accuracy'][-1]:.4f}"
        )


# ============================================================
# Блок 5 – Инференс: загрузка изображения и предсказание
# ============================================================
st.header("🔮 Распознавание загруженной цифры")
uploaded_file = st.file_uploader(
    "Выберите изображение цифры (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Загруженное изображение", width=150)

    img = image.resize((28, 28))
    img_array = np.array(img)

    # MNIST: белая цифра на чёрном фоне. Если фон светлый — инвертируем.
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        st.info(
            "🔄 Изображение инвертировано "
            "(приведено к стилю MNIST: чёрный фон, белая цифра)"
        )

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    model = st.session_state["model"]
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    st.subheader(f"🧾 Результат: **Цифра {predicted_class}**")
    st.write(f"Уверенность: **{confidence:.4f}**")
    st.bar_chart(predictions)


# ============================================================
# Блок 6 – Информация о модели
# ============================================================
with st.expander("ℹ️ Текущая модель"):
    st.write("Архитектура:")
    st.session_state["model"].summary(print_fn=lambda x: st.text(x))
    if "history" in st.session_state:
        st.write("История обучения (последняя эпоха):")
        st.json(
            {k: float(v[-1]) for k, v in st.session_state["history"].history.items()}
        )