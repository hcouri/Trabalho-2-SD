import os
import glob
import time
import pandas as pd
import numpy as np

def carregar_dados_3w(diretorio_0):
    """
    Lê todos os arquivos Parquet no diretório de entrada e concatena
    as colunas numéricas em uma única matriz (DataFrame).
    """
    arquivos = glob.glob(os.path.join(diretorio_0, "*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo Parquet encontrado em: {os.path.abspath(diretorio_0)}")
    
    lista_dfs = []
    for f in arquivos:
        df = pd.read_parquet(f)
        lista_dfs.append(df)
    
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    
    # Seleciona apenas colunas numéricas (ignorando timestamps e rótulos)
    colunas_numericas = df_completo.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove rótulos de classe se presentes
    colunas_sensores = [col for col in colunas_numericas if col.lower() != 'class']
    
    return df_completo[colunas_sensores]

def calcular_estatisticas_referencia(df):
    """
    Calcula Média, Mínimo, Máximo e Desvio Padrão para cada coluna.
    """
    resultados = {}
    for col in df.columns:
        serie = df[col].dropna()  # Remove valores ausentes (NaN)
        resultados[col] = {
            "media": float(serie.mean()),
            "minimo": float(serie.min()),
            "maximo": float(serie.max()),
            "desvio_padrao": float(serie.std(ddof=1))
        }
    return resultados

if __name__ == "__main__":
    # Caminho base tentando encontrar a pasta 3W no diretório atual ou no diretório pai
    candidatos_caminho = [
        os.path.join("..", "3W", "dataset", "0"),
        os.path.join("3W", "dataset", "0"),
        os.path.join("..", "..", "3W", "dataset", "0")
    ]
    
    caminho_dataset = None
    for caminho in candidatos_caminho:
        if os.path.exists(caminho) and len(glob.glob(os.path.join(caminho, "*.parquet"))) > 0:
            caminho_dataset = caminho
            break

    if caminho_dataset is None:
        raise FileNotFoundError("Não foi possível localizar a pasta '3W/dataset/0' contendo arquivos Parquet.")

    print(f"Lendo arquivos Parquet de: {os.path.abspath(caminho_dataset)}...")
    inicio = time.time()
    
    df_dados = carregar_dados_3w(caminho_dataset)
    
    print(f"Matriz carregada com sucesso! Dimensões: {df_dados.shape[0]} linhas x {df_dados.shape[1]} colunas.")
    print(f"Colunas numéricas identificadas: {list(df_dados.columns)}")
    
    # Cálculo das estatísticas
    estatisticas = calcular_estatisticas_referencia(df_dados)
    fim = time.time()
    
    tempo_execucao = fim - inicio
    print(f"\n--- Resultados de Referência (Sequencial) ---")
    print(f"Tempo total de execução: {tempo_execucao:.4f} segundos\n")
    
    # Formatação limpa dos resultados
    df_res = pd.DataFrame(estatisticas).T
    print(df_res.to_string())

    # Salva o gabarito para verificação nos testes automatizados
    os.makedirs("results", exist_ok=True)
    df_res.to_csv(os.path.join("results", "gabarito_referencia.csv"))
    print("\nResultados salvos em 'results/gabarito_referencia.csv' para validação das Pessoas 2 e 3.")