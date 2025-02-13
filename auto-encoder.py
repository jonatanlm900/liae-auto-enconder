import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Carregar os dados
file_path = "/home/johan/Desktop/10_Profissional/dataset/low.pkl"  # Ajuste conforme necessário
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Verificar e padronizar os tamanhos dos arrays
max_length = max(len(arr) for row in data['x'] for arr in row)

def pad_arrays(row, max_length):
    """Padroniza os arrays para o mesmo tamanho preenchendo com zeros."""
    return np.hstack([np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in row])

x_list = np.array([pad_arrays(row, max_length) for row in data['x']], dtype=np.float32)

# Dividir em treino e teste
split = int(0.8 * len(x_list))
x_train, x_test = x_list[:split], x_list[split:]

# Definir o modelo do Autoencoder
input_dim = x_train.shape[1]  # Número de features ajustado

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Treinar o modelo
history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=16, shuffle=True, validation_data=(x_test, x_test))

# Fazer previsões
decoded_imgs = autoencoder.predict(x_test)

# Plotar os resultados
n = 5  # Número de amostras para visualizar
plt.figure(figsize=(10, 4))
for i in range(n):
    # Entrada original
    plt.subplot(2, n, i + 1)
    plt.plot(x_test[i])
    plt.title("Original")
    plt.axis("off")

    # Saída reconstruída
    plt.subplot(2, n, i + 1 + n)
    plt.plot(decoded_imgs[i])
    plt.title("Reconstruído")
    plt.axis("off")

plt.tight_layout()
plt.savefig("reconstrucao.png")  # Salva o gráfico