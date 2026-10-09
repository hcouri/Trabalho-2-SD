import pandas as pd
import numpy as np
import sys
import os

def validar_resultados(caminho_gabarito, caminho_resultado_paralelo):
    if not os.path.exists(caminho_gabarito):
        print(f"Erro: Gabarito não encontrado em {caminho_gabarito}")
        return False
        
    if not os.path.exists(caminho_resultado_paralelo):
        print(f"Erro: Arquivo do teste paralelo não encontrado em {caminho_resultado_paralelo}")
        return False

    df_gabarito = pd.read_csv(caminho_gabarito, index_col=0)
    df_paralelo = pd.read_csv(caminho_resultado_paralelo, index_col=0)

    # Compara ignorando pequenas diferenças de precisão decimal (ponto flutuante)
    passou = True
    for col in ['media', 'minimo', 'maximo', 'desvio_padrao']:
        if col in df_gabarito.columns and col in df_paralelo.columns:
            # Compara valores ignorando NaNs
            diff = np.abs(df_gabarito[col] - df_paralelo[col])
            max_diff = np.nanmax(diff)
            
            if max_diff > 1e-4:
                print(f" Falha na métrica '{col}': Maior diferença foi {max_diff}")
                passou = False
            else:
                print(f" Métrica '{col}' validada com sucesso! (Diferença máx: {max_diff:.2e})")

    if passou:
        print("\n PARABÉNS! O código paralelo gerou resultados 100% idênticos ao sequencial.")
    else:
        print("\n ATENÇÃO: Há divergências nos resultados paralelos em relação à referência.")
        
    return passou

if __name__ == "__main__":
    gabarito = os.path.join("results", "gabarito_referencia.csv")
    
    # Recebe o caminho do CSV do MPI ou Spark via argumento de linha de comando
    if len(sys.argv) > 1:
        resultado = sys.argv[1]
    else:
        resultado = os.path.join("results", "resultado_teste.csv")
        
    validar_resultados(gabarito, resultado)