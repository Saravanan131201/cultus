# ============================
# Advanced Time Series Forecasting with LSTM + Explainability
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# 1. Generate Multivariate Dataset
# -----------------------------
np.random.seed(42)

n = 800
t = np.arange(n)

trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 50)

feature1 = trend + seasonal + np.random.normal(0, 2, n)
feature2 = 0.5 * trend + 5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 1.5, n)
target = 0.7 * feature1 + 0.3 * feature2 + np.random.normal(0, 1, n)

df = pd.DataFrame({
    "feature1": feature1,
    "feature2": feature2,
    "target": target
})

# -----------------------------
# 2. Scaling & Sequence Creation
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = create_sequences(scaled, SEQ_LEN)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# 3. LSTM Seq2Seq Model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 2)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 4. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print("\nEvaluation Metrics")
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)

# -----------------------------
# 5. Integrated Gradients
# -----------------------------
@tf.function
def integrated_gradients(model, inputs, baseline, steps=50):
    interpolated = [baseline + (float(i)/steps)*(inputs-baseline) for i in range(steps+1)]
    interpolated = tf.convert_to_tensor(interpolated)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated)

    grads = tape.gradient(preds, interpolated)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated = (inputs - baseline) * avg_grads
    return integrated

sample = tf.convert_to_tensor(X_test[0:1], dtype=tf.float32)
baseline = tf.zeros_like(sample)

ig = integrated_gradients(model, sample, baseline)

# -----------------------------
# 6. Feature Importance
# -----------------------------
feature_importance = tf.reduce_mean(tf.abs(ig), axis=1).numpy()[0]

plt.figure()
plt.bar(["feature1", "feature2"], feature_importance)
plt.title("Feature Importance (Integrated Gradients)")
plt.show()

# -----------------------------
# 7. Temporal Lag Importance
# -----------------------------
lag_importance = np.mean(np.abs(ig.numpy()), axis=2)[0]

plt.figure()
plt.plot(lag_importance)
plt.title("Temporal Lag Importance")
plt.xlabel("Time Step")
plt.ylabel("Attribution")
plt.show()
