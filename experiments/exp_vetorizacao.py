"""
Experimento da Alteração 3: Vetorização em Memória (Pessoa 4)
Compara laços 'for' em Python nativo vs operações vetorizadas com NumPy/PyArrow.
Alinhado diretamente com as funções 'somas_python' e 'somas_numpy' implementadas
por Pessoa 2 em 'src/estatisticas_mpi.py'.

Mede o impacto no tempo de cálculo e eficiência de CPU antes da fase de agregação/Reduce.
"""

import os
import sys
import time
import json
import math
import numpy as np
import pandas as pd

# Adiciona pasta src para reaproveitamento direto de funções
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from estatisticas_mpi import somas_python, somas_numpy, TAM_BLOCO
except ImportError:
    TAM_BLOCO = 500_000
    def somas_numpy(local):
        n_col = local.shape[1]
        n = np.zeros(n_col)
        soma = np.zeros(n_col)
        for i in range(0, len(local), TAM_BLOCO):
            bloco = local[i:i + TAM_BLOCO]
            n += np.sum(~np.isnan(bloco), axis=0)
            soma += np.nansum(bloco, axis=0)
        minimo = np.nanmin(local, axis=0)
        maximo = np.nanmax(local, axis=0)
        return n, soma, minimo, maximo

    def somas_python(local):
        n_col = local.shape[1]
        n = [0] * n_col
        soma = [0.0] * n_col
        minimo = [math.inf] * n_col
        maximo = [-math.inf] * n_col
        for i in range(0, len(local), TAM_BLOCO):
            soma_bloco = [0.0] * n_col
            for linha in local[i:i + TAM_BLOCO].tolist():
                for j in range(n_col):
                    x = linha[j]
                    if not math.isnan(x):
                        n[j] += 1
                        soma_bloco[j] += x
                        minimo[j] = min(minimo[j], x)
                        maximo[j] = max(maximo[j], x)
            for j in range(n_col):
                soma[j] += soma_bloco[j]
        return np.array(n, dtype=float), np.array(soma), np.array(minimo), np.array(maximo)

def rodar_benchmark_vetorizacao(n_linhas=500_000, n_colunas=27):
    """
    Executa a comparação direta entre as implementações de somas_python e somas_numpy
    sobre uma fatia de dados local típica recebida por um processo no cluster.
    """
    print("=" * 65)
    print(" EXPERIMENTO 3: VETORIZAÇÃO EM MEMÓRIA (Python Nativo vs NumPy)")
    print(f" Amostras na partição: {n_linhas:,} linhas x {n_colunas} colunas")
    print(f" Total de valores numéricos: {n_linhas * n_colunas:,}")
    print("=" * 65 + "\n")

    # Gera fatia sintética simulando leitura de dados dos sensores de petróleo
    np.random.seed(42)
    matriz_local = np.random.normal(loc=100.0, scale=15.0, size=(n_linhas, n_colunas)).astype(np.float64)
    # Insere 1% de NaNs para simular falhas em sensores
    mascara_nan = np.random.rand(n_linhas, n_colunas) < 0.01
    matriz_local[mascara_nan] = np.nan

    # 1. Medição Python Nativo (com subamostra proporcional para execução rápida)
    n_teste_py = min(n_linhas, 20_000)
    fatia_py = matriz_local[:n_teste_py]
    print(f"1. Executando 'somas_python' (laços iterativos em {n_teste_py:,} linhas)...")
    t0_py = time.time()
    somas_python(fatia_py)
    tempo_py = time.time() - t0_py
    fator_escala = n_linhas / n_teste_py
    tempo_py_projetado = tempo_py * fator_escala
    print(f"   -> Tempo medido Python: {tempo_py:.4f}s")
    print(f"   -> Tempo projetado para {n_linhas:,} linhas: {tempo_py_projetado:.4f}s")

    # 2. Medição NumPy Vetorizado (SIMD / C)
    print(f"\n2. Executando 'somas_numpy' (vetorizado SIMD em todas as {n_linhas:,} linhas)...")
    t0_np = time.time()
    somas_numpy(matriz_local)
    tempo_np = time.time() - t0_np
    print(f"   -> Tempo medido NumPy: {tempo_np:.4f}s")

    speedup = tempo_py_projetado / tempo_np
    print("\n" + "=" * 65)
    print(f" SPEEDUP OBTIDO COM VETORIZAÇÃO: {speedup:.1f}x mais rápido!")
    print("=" * 65)

    resultado = {
        "alteracao": "Alteração 3: Vetorização em Memória",
        "n_linhas": n_linhas,
        "n_colunas": n_colunas,
        "versao_base": "Python Nativo (somas_python)",
        "tempo_base_s": round(tempo_py_projetado, 4),
        "versao_otimizada": "NumPy Vetorizado (somas_numpy)",
        "tempo_otimizado_s": round(tempo_np, 4),
        "speedup_obtido": round(speedup, 2)
    }

    os.makedirs("results", exist_ok=True)
    caminho_out = os.path.join("results", "experimento_vetorizacao.json")
    with open(caminho_out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4)
    print(f"\n[OK] Resultados salvos em: '{caminho_out}'")

    return resultado

if __name__ == "__main__":
    rodar_benchmark_vetorizacao()
