"""
Módulo de Cálculo de Métricas de Desempenho e Escalabilidade (Pessoa 4)
Calcula Speedup S(p) = T1 / Tp e Eficiência E(p) = S(p) / p
conforme a metodologia acadêmica de avaliação de sistemas distribuídos (Saxena, 2020).
"""

import os
import json
import pandas as pd
import numpy as np

def carregar_meta_referencia(caminho_meta=os.path.join("results", "meta_referencia.json")):
    """
    Carrega o dicionário completo de metadados gerados pela Pessoa 1,
    incluindo tempo de referência sequencial T1, volume de dados e especificações de hardware.
    """
    if os.path.exists(caminho_meta):
        try:
            with open(caminho_meta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[AVISO] Não foi possível ler {caminho_meta}: {e}")
            
    # Valores padrão de fallback
    return {
        "categorias_processadas": ["0"],
        "linhas_totais": 12158183,
        "colunas_sensores": 27,
        "tempo_carregamento_s": 9.4893,
        "tempo_calculo_s": 4.8368,
        "tempo_total_t1_s": 14.3292,
        "ambiente_hardware": {
            "processador": "Intel Core i5-11400F (6 cores / 12 threads)",
            "placa_de_video": "NVIDIA GeForce RTX 4060 8GB",
            "memoria_ram": "8 GB DDR4",
            "sistema_operacional": "Windows 11"
        }
    }

def carregar_t1_referencia(caminho_meta=os.path.join("results", "meta_referencia.json")):
    """
    Retorna o tempo oficial de execução sequencial T1 (em segundos).
    """
    meta = carregar_meta_referencia(caminho_meta)
    return float(meta.get("tempo_total_t1_s", 14.3292))

def calcular_speedup_eficiencia(df_tempos, t1=None):
    """
    Recebe um DataFrame com colunas ['tecnologia', 'p', 'tempo_medio_s']
    e calcula Speedup S(p) e Eficiência E(p).
    
    Fórmulas Acadêmicas:
      S(p) = T1 / Tp
      E(p) = S(p) / p  (ou percentual: E(p) * 100)
    """
    if t1 is None:
        t1 = carregar_t1_referencia()
        
    df = df_tempos.copy()
    
    # Speedup: S(p) = T1 / Tp
    df['speedup'] = t1 / df['tempo_medio_s']
    
    # Speedup Ideal: S_ideal = p
    df['speedup_ideal'] = df['p'].astype(float)
    
    # Eficiência: E(p) = S(p) / p
    df['eficiencia'] = df['speedup'] / df['p']
    df['eficiencia_percentual'] = df['eficiencia'] * 100.0
    
    # Eficiência Ideal: 100%
    df['eficiencia_ideal_pct'] = 100.0
    
    return df

def importar_tempos_mpi(caminho_tempos=os.path.join("results", "tempos_mpi.csv"), t1=None):
    """
    Importa diretamente as medições geradas pelo script 'src/estatisticas_mpi.py' (Pessoa 2)
    e computa as métricas de speedup e eficiência.
    """
    if not os.path.exists(caminho_tempos):
        return None
        
    df_mpi_raw = pd.read_csv(caminho_tempos)
    # Agrupa por modo, cálculo e p para obter médias caso haja múltiplas rodadas
    agrupado = df_mpi_raw.groupby(['modo', 'calculo', 'p']).agg({
        't_total_s': ['mean', 'std'],
        't_leitura_s': 'mean',
        't_distribuicao_s': 'mean',
        't_calculo_reducao_s': 'mean'
    }).reset_index()
    
    agrupado.columns = ['modo', 'calculo', 'p', 'tempo_medio_s', 'desvio_tempo_s',
                        't_leitura_medio_s', 't_distribuicao_medio_s', 't_calculo_reducao_medio_s']
    
    agrupado['tecnologia'] = agrupado.apply(
        lambda r: f"MPI ({r['modo']}, {r['calculo']})", axis=1
    )
    
    return calcular_speedup_eficiencia(agrupado, t1=t1)

def gerar_tabela_markdown(df_metricas):
    """
    Gera uma tabela formatada em Markdown com as métricas consolidadas.
    """
    cols_format = ['tecnologia', 'p', 'tempo_medio_s', 'speedup', 'eficiencia_percentual']
    df_show = df_metricas[[c for c in cols_format if c in df_metricas.columns]].copy()
    
    df_show['tempo_medio_s'] = df_show['tempo_medio_s'].apply(lambda x: f"{x:.4f}s")
    df_show['speedup'] = df_show['speedup'].apply(lambda x: f"{x:.2f}x")
    df_show['eficiencia_percentual'] = df_show['eficiencia_percentual'].apply(lambda x: f"{x:.1f}%")
    
    df_show.rename(columns={
        'tecnologia': 'Tecnologia',
        'p': 'Processos/Cores (p)',
        'tempo_medio_s': 'Tempo Médio (Tp)',
        'speedup': 'Speedup S(p)',
        'eficiencia_percentual': 'Eficiência E(p)'
    }, inplace=True)
    
    return df_show.to_markdown(index=False)

if __name__ == "__main__":
    meta = carregar_meta_referencia()
    print(f"Tempo de Referência Sequencial Oficial T1: {meta['tempo_total_t1_s']:.4f}s")
    print(f"Ambiente de Hardware: {meta.get('ambiente_hardware', {})}")
