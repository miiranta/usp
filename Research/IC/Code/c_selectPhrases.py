# Thanks to the contribution of Mateus Machado!!!
# https://github.com/mtarcinalli

import os
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

tqdm.pandas()

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences_selected")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
WORD_TO_SEARCH = "inflação"
THRESHOLD_GLOBAL = 0.6 # %

df = pd.DataFrame()
filtered_df = pd.DataFrame()

# Função para calcular a distância (cosine similarity invertida)
def calcular_distancia(sentenca, vetor_referencia, embeddings):
    vetor_sentenca = embeddings.embed_query(sentenca)
    # Cosine similarity retorna valor entre -1 e 1; quanto mais próximo de 1, mais similar
    similaridade = cosine_similarity([vetor_sentenca], [vetor_referencia])[0][0]
    # Para transformar em "distância", usamos 1 - similaridade
    distancia = 1 - similaridade
    return distancia

def vectorize():
    dados = []
    
    # Loading dataset
    for nome_arquivo in os.listdir(INPUT_FOLDER):
        caminho = os.path.join(INPUT_FOLDER, nome_arquivo)
        if os.path.isfile(caminho):
            data = nome_arquivo.replace(".txt", "")  # remove extensão se houver
            with open(caminho, "r", encoding="utf-8") as f:
                for linha in f:
                    sentenca = linha.strip()
                    if sentenca:
                        dados.append({"data": data, "sentenca": sentenca})
                        
    global df
    df = pd.DataFrame(dados)
    df["data"] = pd.to_datetime(df["data"], format="%d%m%Y", errors="coerce")
    
    print('Quantidade inicial de sentenças:', len(df))

    # 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': device}
    )
    
    # 
    vetor_inflacao = embeddings.embed_query(WORD_TO_SEARCH)

    #
    df["inflation"] = df["sentenca"].progress_apply(lambda x: calcular_distancia(x, vetor_inflacao, embeddings))

    #
    #df.to_csv(os.path.join(OUTPUT_FOLDER, "tmp.csv"), index=False)
    
def select():
    global filtered_df
    filtered_df = df[(df['inflation'] < THRESHOLD_GLOBAL)]
      
    print('Quantidade final de sentenças:', len(filtered_df),'\n')
      
def save():
    grouped_df = filtered_df.groupby(filtered_df['data'].dt.strftime('%d%m%Y'))
    
    for date, group in grouped_df:
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{date}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for sentence in group['sentenca']:
                output_file.write(sentence + "\n")

def main():
    vectorize()
    select()
    save()

if __name__ == "__main__":
    main()