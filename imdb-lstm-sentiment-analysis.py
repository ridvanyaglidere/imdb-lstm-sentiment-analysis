import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# -----------------------------
# 1. Veri Yükleme
# -----------------------------
max_features = 100000  # En sık kullanılan 100k kelime
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# -----------------------------
# 2. Pad İşlemi
# -----------------------------
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# -----------------------------
# 3. Model Kurulumu
# -----------------------------
def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=64, input_length=maxlen))
    model.add(LSTM(units=8))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation="sigmoid"))  # Binary sınıf

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_lstm_model()
model.summary()

# -----------------------------
# 4. Early Stopping
# -----------------------------
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# 5. Model Eğitimi
# -----------------------------
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# -----------------------------
# 6. Değerlendirme
# -----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# -----------------------------
# 7. Grafik Çizimi
# -----------------------------
plt.figure(figsize=(12, 5))

# Loss grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.grid(True)

# Accuracy grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()
