import os
import glob
import time
import argparse
import json
import pandas as pd
import numpy as np

def carregar_dados_3w(diretorio_dataset, categoria="0"):
    """
    Lê os arquivos Parquet preservando a estrutura de séries temporais.
    """
    if categoria.lower() == "todas":
        padrao_busca = os.path.join(diretorio_dataset, "*", "*.parquet")
    else:
        padrao_busca = os.path.join(diretorio_dataset, categoria, "*.parquet")
        
    arquivos = sorted(glob.glob(padrao_busca))
    
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo Parquet encontrado em: {padrao_busca}")
    
    print(f"Total de arquivos identificados: {len(arquivos)}")
    
    lista_dfs = []
    for f in arquivos:
        df = pd.read_parquet(f)
        lista_dfs.append(df)
    
    # Concatena preservando a ordem cronológica dos arquivos
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    
    # Ordenação temporal (caso exista coluna de timestamp explícita)
    colunas_tempo = [col for col in df_completo.columns if 'time' in col.lower() or 'data' in col.lower()]
    if colunas_tempo:
        col_tempo = colunas_tempo[0]
        print(f"Ordenando dataset pela coluna temporal: '{col_tempo}'...")
        df_completo = df_completo.sort_values(by=col_tempo).reset_index(drop=True)
    
    colunas_numericas = df_completo.select_dtypes(include=[np.number]).columns.tolist()
    colunas_sensores = [col for col in colunas_numericas if col.lower() not in ['class', 'state']]
    
    return df_completo[colunas_sensores]

def calcular_estatisticas_referencia(df):
    """
    Calcula N (amostras temporais válidas), média, mínimo, máximo e desvio padrão.
    """
    estatisticas = {}
    
    for col in df.columns:
        serie = df[col].dropna()
        n_amostras = len(serie)
        
        if n_amostras > 0:
            estatisticas[col] = {
                'n_amostras': int(n_amostras),
                'media': float(serie.mean()),
                'minimo': float(serie.min()),
                'maximo': float(serie.max()),
                'desvio_padrao': float(serie.std(ddof=1))
            }
        else:
            estatisticas[col] = {
                'n_amostras': 0,
                'media': np.nan,
                'minimo': np.nan,
                'maximo': np.nan,
                'desvio_padrao': np.nan
            }
            
    df_res = pd.DataFrame.from_dict(estatisticas, orient='index')
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processamento de Referência Sequencial - Dataset 3W")
    parser.add_argument("--categoria", type=str, default="0", help="Categoria do 3W (ex: '0' ou 'todas')")
    args = parser.parse_args()

    candidatos_caminho = [
        os.path.join("..", "3W", "dataset"),
        os.path.join("3W", "dataset"),
        os.path.join("..", "..", "3W", "dataset")
    ]
    
    caminho_dataset = None
    for caminho in candidatos_caminho:
        if os.path.exists(caminho):
            caminho_dataset = caminho
            break

    if not caminho_dataset:
        raise FileNotFoundError("Pasta '3W/dataset' não encontrada nos caminhos padrão.")

    print(f"Lendo categoria [{args.categoria}] de: {os.path.abspath(caminho_dataset)}...")
    inicio = time.time()
    
    df_dados = carregar_dados_3w(caminho_dataset, categoria=args.categoria)
    
    tempo_carregamento = time.time() - inicio
    print(f"Matriz carregada! Dimensões: {df_dados.shape[0]} instantes temporais x {df_dados.shape[1]} sensores.")
    
    inicio_calculo = time.time()
    df_gabarito = calcular_estatisticas_referencia(df_dados)
    tempo_calculo = time.time() - inicio_calculo
    
    tempo_total = time.time() - inicio
    
    os.makedirs("results", exist_ok=True)
    caminho_csv = os.path.join("results", "gabarito_referencia.csv")
    df_gabarito.to_csv(caminho_csv)
    
    meta_info = {
        "categoria_processada": args.categoria,
        "linhas_totais": int(df_dados.shape[0]),
        "colunas_totais": int(df_dados.shape[1]),
        "tempo_carregamento_segundos": round(tempo_carregamento, 4),
        "tempo_calculo_segundos": round(tempo_calculo, 4),
        "tempo_total_t1_segundos": round(tempo_total, 4)
    }
    
    with open(os.path.join("results", "meta_referencia.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)

    print(f"\n--- Resultados de Referência (Sequencial) ---")
    print(f"Tempo total de execução (T1): {tempo_total:.4f} segundos")
    print(df_gabarito)
    print(f"\nGabarito salvo em '{caminho_csv}' e metadados em 'results/meta_referencia.json'.")