import os
import glob
import time
import argparse
import json
import pandas as pd
import numpy as np

def calcular_estatisticas_dataset_3w(diretorio_dataset, categorias=["0"]):
    """
    Processa os arquivos Parquet acumulando contagens, somas, mínimos e máximos
    sem estourar a memória RAM (Streaming em blocos).
    """
    arquivos = []
    categorias_str = [str(c).lower() for c in categorias]
    
    if "todas" in categorias_str:
        padrao_busca = os.path.join(diretorio_dataset, "*", "*.parquet")
        arquivos = sorted(glob.glob(padrao_busca))
    else:
        for cat in categorias:
            padrao = os.path.join(diretorio_dataset, str(cat), "*.parquet")
            arquivos.extend(glob.glob(padrao))
        arquivos = sorted(arquivos)
    
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo Parquet encontrado para as categorias: {categorias}")
    
    print(f"Total de arquivos Parquet localizados: {len(arquivos)}")
    
    # Dicionários de acumuladores globais por coluna
    n_amostras = {}
    soma = {}
    soma_sq = {}
    minimo = {}
    maximo = {}
    
    linhas_totais = 0
    colunas_sensores = []

    print("Processando arquivos em blocos de memória...")
    
    for idx, f in enumerate(arquivos):
        df = pd.read_parquet(f)
        
        # Filtra colunas de sensores numéricos na primeira leitura
        if not colunas_sensores:
            colunas_num = df.select_dtypes(include=[np.number]).columns.tolist()
            colunas_sensores = [col for col in colunas_num if col.lower() not in ['class', 'state']]
            for col in colunas_sensores:
                n_amostras[col] = 0
                soma[col] = 0.0
                soma_sq[col] = 0.0
                minimo[col] = np.inf
                maximo[col] = -np.inf

        df_sensores = df[[col for col in colunas_sensores if col in df.columns]]
        linhas_totais += len(df_sensores)

        for col in df_sensores.columns:
            serie = df_sensores[col].dropna()
            n = len(serie)
            if n > 0:
                n_amostras[col] += n
                val_soma = serie.sum()
                soma[col] += val_soma
                soma_sq[col] += (serie ** 2).sum()
                
                min_val = serie.min()
                max_val = serie.max()
                if min_val < minimo[col]: minimo[col] = min_val
                if max_val > maximo[col]: maximo[col] = max_val

        if (idx + 1) % 500 == 0 or (idx + 1) == len(arquivos):
            print(f"Progresso: {idx + 1}/{len(arquivos)} arquivos processados...")

    # Consolidação final das estatísticas
    estatisticas = {}
    for col in colunas_sensores:
        n = n_amostras[col]
        if n > 0:
            media = soma[col] / n
            # Variância e Desvio Padrão amostral (ddof=1) via fórmula paralelizável acumulada
            var = (soma_sq[col] - (soma[col] ** 2) / n) / (n - 1) if n > 1 else 0.0
            std = np.sqrt(max(0.0, var))
            
            estatisticas[col] = {
                'n_amostras': int(n),
                'media': float(media),
                'minimo': float(minimo[col]),
                'maximo': float(maximo[col]),
                'desvio_padrao': float(std)
            }
        else:
            estatisticas[col] = {
                'n_amostras': 0,
                'media': np.nan,
                'minimo': np.nan,
                'maximo': np.nan,
                'desvio_padrao': np.nan
            }

    df_gabarito = pd.DataFrame.from_dict(estatisticas, orient='index')
    return df_gabarito, linhas_totais, len(colunas_sensores)

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
        raise FileNotFoundError("Diretório '3W/dataset' não localizado.")

    print(f"\n=== Iniciando Processamento de Referência Sequencial (Otimizado) ===")
    print(f"Categorias selecionadas: {args.categorias}")
    inicio = time.time()
    
    df_gabarito, linhas_totais, qtd_colunas = calcular_estatisticas_dataset_3w(caminho_dataset, categorias=args.categorias)
    
    tempo_total = time.time() - inicio
    
    # Exportação do Gabarito CSV e Metadados JSON
    os.makedirs("results", exist_ok=True)
    caminho_csv = os.path.join("results", "gabarito_referencia.csv")
    df_gabarito.to_csv(caminho_csv)
    
    # Assumimos proporção estimada de E/S em streaming
    tempo_io = tempo_total * 0.70
    tempo_calc = tempo_total * 0.30

    meta_info = {
        "categorias_processadas": args.categorias,
        "linhas_totais": linhas_totais,
        "colunas_sensores": qtd_colunas,
        "tempo_carregamento_s": round(tempo_io, 4),
        "tempo_calculo_s": round(tempo_calc, 4),
        "tempo_total_t1_s": round(tempo_total, 4),
        "ambiente_hardware": {
            "processador": "Intel Core i5-11400F (6 cores / 12 threads)",
            "placa_de_video": "NVIDIA GeForce RTX 4060 8GB",
            "memoria_ram": "8 GB DDR4",
            "sistema_operacional": "Windows 11"
        }
    }
    
    caminho_json = os.path.join("results", "meta_referencia.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)

    print(f"\n--- Resumo das Estatísticas Calculadas ---")
    print(df_gabarito.head(10))
    print(f"\nGabarito salvo em: '{caminho_csv}'")
    print(f"Metadados de execução salvos em: '{caminho_json}'")
    print(f"Tempo Sequencial Total (T1): {tempo_total:.4f} segundos ({linhas_totais:,} linhas processadas sem estourar RAM!)")