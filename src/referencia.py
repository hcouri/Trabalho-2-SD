import os
import glob
import time
import argparse
import json
import pandas as pd
import numpy as np

def carregar_dados_3w(diretorio_dataset, categorias=["0"]):
    """
    Lê os arquivos Parquet de uma ou mais categorias especificadas (ex: ['0', '3', '5'] ou ['todas']).
    Preserva a ordem cronológica e filtra apenas as colunas de sensores.
    """
    arquivos = []
    categorias_str = [str(c).lower() for c in categorias]
    
    if "todas" in categorias_str:
        padrao_busca = os.path.join(diretorio_dataset, "*", "*.parquet")
        arquivos = sorted(glob.glob(padrao_busca))
    else:
        for cat in categorias:
            padrao = os.path.join(diretorio_dataset, str(cat), "*.parquet")
            encontrados = glob.glob(padrao)
            arquivos.extend(encontrados)
        arquivos = sorted(arquivos)
    
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo Parquet encontrado para as categorias: {categorias}")
    
    print(f"Total de arquivos Parquet localizados para as categorias {categorias}: {len(arquivos)}")
    
    lista_dfs = []
    for f in arquivos:
        df = pd.read_parquet(f)
        lista_dfs.append(df)
    
    # Concatena mantendo a integridade cronológica dos arquivos
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    
    # Ordenação temporal caso exista coluna explícita
    colunas_tempo = [col for col in df_completo.columns if 'time' in col.lower() or 'data' in col.lower()]
    if colunas_tempo:
        col_tempo = colunas_tempo[0]
        print(f"Ordenando cronologicamente pela coluna: '{col_tempo}'...")
        df_completo = df_completo.sort_values(by=col_tempo).reset_index(drop=True)
    
    # Seleciona apenas colunas de sensores numéricos (descarte de colunas administrativas/metadados)
    colunas_numericas = df_completo.select_dtypes(include=[np.number]).columns.tolist()
    colunas_sensores = [col for col in colunas_numericas if col.lower() not in ['class', 'state']]
    
    return df_completo[colunas_sensores]

def calcular_estatisticas_referencia(df):
    """
    Calcula N (amostras temporais válidas), Média, Mínimo, Máximo e Desvio Padrão amostral (ddof=1).
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
            
    return pd.DataFrame.from_dict(estatisticas, orient='index')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gabarito Sequencial de Referência - Dataset 3W")
    parser.add_argument("--categorias", nargs="+", default=["0"], help="Categorias a processar (ex: --categorias 0 3 5 ou --categorias todas)")
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
        raise FileNotFoundError("Diretório '3W/dataset' não foi encontrado nos caminhos padrão.")

    print(f"\n=== Iniciando Processamento de Referência Sequencial ===")
    print(f"Categorias selecionadas: {args.categorias}")
    inicio = time.time()
    
    df_dados = carregar_dados_3w(caminho_dataset, categorias=args.categorias)
    tempo_carregamento = time.time() - inicio
    
    print(f"Matriz Carregada: {df_dados.shape[0]:,} linhas x {df_dados.shape[1]} colunas de sensores.")
    
    inicio_calculo = time.time()
    df_gabarito = calcular_estatisticas_referencia(df_dados)
    tempo_calculo = time.time() - inicio_calculo
    
    tempo_total = time.time() - inicio
    
    # Exportação do Gabarito CSV e Metadados JSON
    os.makedirs("results", exist_ok=True)
    caminho_csv = os.path.join("results", "gabarito_referencia.csv")
    df_gabarito.to_csv(caminho_csv)
    
    meta_info = {
        "categorias_processadas": args.categorias,
        "linhas_totais": int(df_dados.shape[0]),
        "colunas_sensores": int(df_dados.shape[1]),
        "tempo_carregamento_s": round(tempo_carregamento, 4),
        "tempo_calculo_s": round(tempo_calculo, 4),
        "tempo_total_t1_s": round(tempo_total, 4)
    }
    
    caminho_json = os.path.join("results", "meta_referencia.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)

    print(f"\n--- Resumo das Estatísticas Calculadas ---")
    print(df_gabarito.head(10))
    print(f"\nGabarito salvo em: '{caminho_csv}'")
    print(f"Metadados de execução salvos em: '{caminho_json}'")
    print(f"⏱Tempo Sequencial Total (T1): {tempo_total:.4f} segundos")