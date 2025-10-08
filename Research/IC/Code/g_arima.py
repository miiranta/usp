# Code by Cezio
# https://github.com/clfjunior

# Adapted by Lucas Miranda

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Carregar os dados da série temporal
def load_series():
    file_path = 'C:/Users/cezio/Downloads/ipeadata[06-01-2025-02-01].csv'
    data = pd.read_csv(file_path, sep=';', decimal=',')
    
    # Remover colunas sem utilidade
    data = data.dropna(axis=1, how='all')
    
    # Exiba os nomes das colunas para verificar
    print(data.head())
    print(data.columns)
    
    # Identificar a coluna correta que contém os valores numéricos
    column_name = data.columns[1]  # Supondo que a segunda coluna contém os valores
    
    # Converter os valores para float
    series = data[column_name].astype(float)
    
    # Corrigir o formato da data explicitamente
    series.index = pd.to_datetime(data.iloc[:, 0], format='%Y.%m', errors='coerce')
    series = series.dropna()  # Remover valores inválidos
    
    # Filtrar os dados a partir de novembro de 1993
    series = series[series.index >= '1993-12']
    
    return series

# Dividir os dados em treino e teste
def split_data(series, train_ratio=0.70):
    size = int(len(series) * train_ratio)
    train = series[:size]
    test = series[size:]
    return train, test

# Função principal
if __name__ == "__main__":
    try:
        # Carregando os dados
        series = load_series()

        # Divisão em treino e teste
        train, test = split_data(series)

        # Exibir o número de observações nos conjuntos de treino e teste
        print(f"Número total de observações: {len(series)}")
        print(f"Número de observações para treinamento: {len(train)}")
        print(f"Número de observações para teste: {len(test)}")

        # Preparação da estrutura de dados
        history = list(train)
        predictions = []

        # Parâmetros do ARIMA (ajuste conforme necessário)
        p, d, q = 1, 1, 0  # Exemplo de ordem do modelo

        # Loop para previsão
        for t in range(len(test)):
            # Treinar o modelo ARIMA
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit()

            # Fazer previsão
            hat = model_fit.forecast()[0]
            predictions.append(hat)

            # Atualizar histórico com valor real
            observed = test.iloc[t]
            history.append(observed)

        # Calcular o erro
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mse)

        # Exibir o RMSE
        print(f"RMSE de Teste: {rmse}")

        # Exibir previsões junto com os valores reais
        plt.figure(figsize=(12, 6))

        # Subplot 1: Série Temporal
        plt.subplot(2, 1, 1)
        plt.plot(series.index, series, label='Série Temporal')
        plt.title('Série Temporal')
        plt.legend()

        # Subplot 2: Previsões vs Valores Reais e RMSE
        plt.subplot(2, 1, 2)
        plt.plot(test.index, test, label='Valores Reais')
        plt.plot(test.index, predictions, label='Previsões', color='red')
        plt.title(f'Previsões vs Valores Reais (RMSE: {rmse})')
        plt.legend()

        # Ajustar layout e exibir o gráfico
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print(e)