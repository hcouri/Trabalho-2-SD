import pandas as pd
import numpy as np
import sys
import os

def validar_resultados(caminho_gabarito, caminho_resultado_paralelo, tolerancia=1e-4):
    if not os.path.exists(caminho_gabarito):
        print(f"Erro: Gabarito não encontrado em {caminho_gabarito}")
        return False
        
    if not os.path.exists(caminho_resultado_paralelo):
        print(f"Erro: Arquivo de resultado paralelo não encontrado em {caminho_resultado_paralelo}")
        return False

    df_gabarito = pd.read_csv(caminho_gabarito, index_col=0)
    df_paralelo = pd.read_csv(caminho_resultado_paralelo, index_col=0)

    metricas = ['n_amostras', 'media', 'minimo', 'maximo', 'desvio_padrao']
    passou_geral = True

    print("\n--- Relatório de Validação de Resultados ---")
    for col in metricas:
        if col in df_gabarito.columns and col in df_paralelo.columns:
            # Substitui NaNs para comparação contínua
            v_gab = df_gabarito[col].fillna(np.nan)
            v_par = df_paralelo[col].fillna(np.nan)
            
            diff = np.abs(v_gab - v_par)
            max_diff = np.nanmax(diff) if not np.all(np.isnan(diff)) else 0.0
            
            if max_diff > tolerancia:
                print(f"Métrica '{col:15s}': DIVERGENTE (Diferença máx: {max_diff:.2e})")
                passou_geral = False
            else:
                print(f"Métrica '{col:15s}': OK (Diferença máx: {max_diff:.2e})")
        else:
            print(f"Métrica '{col}' não encontrada em ambos os arquivos.")

    if passou_geral:
        print("\nVALIDAÇÃO SUCEDIDA! A implementação paralela reproduziu fielmente a referência.")
    else:
        print("\nFALHA NA VALIDAÇÃO: Verifique os acumuladores de borda/reduce na versão paralela.")
        
    return passou_geral

if __name__ == "__main__":
    gabarito = os.path.join("results", "gabarito_referencia.csv")
    resultado = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "resultado_teste.csv")
    validar_resultados(gabarito, resultado)