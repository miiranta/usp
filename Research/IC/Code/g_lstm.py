# Code by Cezio
# https://github.com/clfjunior

# Adapted by Lucas Miranda

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt

# Função para dividir os dados em treino e teste
def split_data(series, train_ratio=0.7):
    size = int(len(series) * train_ratio)
    return series[:size], series[size:]

# Função para ajustar o modelo LSTM com EarlyStopping e melhores hiperparâmetros
def fit_lstm(train, epochs, neurons):
    from tensorflow.keras.callbacks import EarlyStopping

    X, y = train[:-1], train[1:]
    X = X.reshape((X.shape[0], 1, 1))
    y = y.reshape((y.shape[0], 1))

    model = Sequential([
        Input(shape=(1, 1)),
        LSTM(neurons),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='Nadam')

    # EarlyStopping para evitar overfitting e melhorar o desempenho
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model.fit(X, y, epochs=epochs, batch_size=1, shuffle=False, verbose=0, callbacks=[early_stopping])
    return model

# Função para fazer uma previsão de um passo à frente
def forecast_lstm(model, X):
    X = X.reshape((1, 1, 1))
    return model.predict(X, batch_size=1, verbose=0)[0, 0]

# Função principal para calcular o RMSE e exibir gráficos
def main():
    # Carregar os dados da série temporal
    series = pd.read_csv('C:/Users/cezio/Downloads/ipeadata[06-01-2025-02-01].csv', header=0, index_col=0, parse_dates=True, date_format='%d/%m/%Y').squeeze()

    # Filtrar dados a partir de dezembro de 1993
    series = series[series.index >= '1993-12-01'].ffill().bfill()

    # Dividir os dados em treino e teste
    train, test = split_data(series.values)

    # Ajustar o modelo LSTM
    lstm_model = fit_lstm(train, epochs=1, neurons=500)

    # Previsões de teste
    test_predictions = [forecast_lstm(lstm_model, np.array([test[i]])) for i in range(len(test))]

    # Calcular RMSE de teste
    test_rmse = sqrt(mean_squared_error(test, test_predictions))

    # Exibir número de observações de treino, de teste e RMSE de teste
    num_train_observations = len(train)
    num_test_observations = len(test)
    print(f'Número de observações de treinamento: {num_train_observations}')
    print(f'Número de observações de teste: {num_test_observations}')
    print(f'RMSE - Teste: {test_rmse:.3f}')

    # Plotar RMSE contra observações
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_test_observations), test, label='Observações reais', color='blue')
    plt.plot(range(num_test_observations), test_predictions, label='Previsões', color='red', linestyle='dashed')
    plt.title('Previsões vs Observações Reais')
    plt.xlabel('Observações')
    plt.ylabel('Valores')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()