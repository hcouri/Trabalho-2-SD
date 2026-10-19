# Estatísticas do dataset 3W com MPI (mpi4py) - Pessoa 2
#
# Calcula N, média, mínimo, máximo e desvio padrão (ddof=1) de cada sensor.
#
# Opções usadas nas alterações de desempenho:
#   --modo centralizado : o Rank 0 lê tudo e envia fatias da matriz com Scatterv (padrão)
#   --modo distribuido  : cada processo lê os seus próprios arquivos Parquet (Alteração 1)
#   --calculo numpy     : cálculo local vetorizado com NumPy (padrão)
#   --calculo python    : cálculo local com laços em Python (Alteração 3)
#
# Execução:
#   mpiexec -n 4 python src/estatisticas_mpi.py --categorias 0
#   mpiexec -n 4 python src/estatisticas_mpi.py --categorias 0 --modo distribuido

import argparse
import glob
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from referencia import carregar_dados_3w  # leitura da Pessoa 1

# np.nanmin/np.nanmax avisam quando uma coluna é toda NaN; isso é esperado aqui
warnings.filterwarnings("ignore", category=RuntimeWarning)

# A fatia local é processada em blocos para não criar cópias enormes na memória
TAM_BLOCO = 500_000


def achar_dataset():
    for caminho in ["../3W/dataset", "3W/dataset", "../../3W/dataset"]:
        if os.path.exists(caminho):
            return caminho
    raise FileNotFoundError("Pasta 3W/dataset não encontrada")


def dividir_linhas(n_linhas, n_colunas, p):
    # Os primeiros 'resto' processos recebem uma linha a mais
    base, resto = divmod(n_linhas, p)
    linhas = np.array([base + 1 if r < resto else base for r in range(p)])
    counts = linhas * n_colunas                       # Scatterv conta elementos, não linhas
    displs = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return linhas, counts, displs


# ---------------- Cálculo local com NumPy ----------------
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


def desvios_numpy(local, media):
    desvios = np.zeros(local.shape[1])
    for i in range(0, len(local), TAM_BLOCO):
        bloco = local[i:i + TAM_BLOCO]
        desvios += np.nansum((bloco - media) ** 2, axis=0)
    return desvios


# ---------------- Cálculo local com laços em Python (Alteração 3) ----------------
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


def desvios_python(local, media):
    n_col = local.shape[1]
    media = media.tolist()
    desvios = [0.0] * n_col
    for i in range(0, len(local), TAM_BLOCO):
        desvios_bloco = [0.0] * n_col
        for linha in local[i:i + TAM_BLOCO].tolist():
            for j in range(n_col):
                x = linha[j]
                if not math.isnan(x):
                    desvios_bloco[j] += (x - media[j]) ** 2
        for j in range(n_col):
            desvios[j] += desvios_bloco[j]
    return np.array(desvios)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categorias", nargs="+", default=["0"])
    parser.add_argument("--modo", choices=["centralizado", "distribuido"], default="centralizado")
    parser.add_argument("--calculo", choices=["numpy", "python"], default="numpy")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    comm.Barrier()
    t_inicio = MPI.Wtime()

    # ===== 1) Leitura e distribuição dos dados =====
    if args.modo == "centralizado":
        # O Rank 0 lê todos os arquivos e monta a matriz
        matriz, colunas = None, None
        if rank == 0:
            df = carregar_dados_3w(achar_dataset(), categorias=args.categorias)
            colunas = df.columns.tolist()
            # O Scatterv envia a memória "linha a linha" (ordem C); o to_numpy pode devolver
            # a matriz guardada coluna a coluna, então garantimos a ordem correta
            matriz = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
            del df
        comm.Barrier()
        t_leitura = MPI.Wtime()

        # Envia o número de linhas e as colunas para todos (objeto pequeno: bcast)
        n_linhas = matriz.shape[0] if rank == 0 else None
        n_linhas, colunas = comm.bcast((n_linhas, colunas), root=0)

        # Envia uma fatia de linhas da matriz para cada processo (buffer grande: Scatterv)
        linhas, counts, displs = dividir_linhas(n_linhas, len(colunas), p)
        local = np.empty((linhas[rank], len(colunas)))
        envio = [matriz, counts, displs, MPI.DOUBLE] if rank == 0 else None
        comm.Scatterv(envio, local, root=0)
        del matriz

    else:
        # Cada processo pega um arquivo a cada p (rodízio) e lê os seus arquivos
        pastas = ["*"] if "todas" in args.categorias else args.categorias
        arquivos = []
        for cat in pastas:
            arquivos += glob.glob(os.path.join(achar_dataset(), cat, "*.parquet"))
        arquivos = sorted(arquivos)
        meus_arquivos = arquivos[rank::p]

        # O Rank 0 define as colunas (mesma regra da referência) e envia para todos.
        # Considera que todos os arquivos do 3W têm as mesmas colunas.
        colunas = None
        if rank == 0:
            exemplo = pd.read_parquet(arquivos[0])
            colunas = [c for c in exemplo.select_dtypes(include=[np.number]).columns
                       if c.lower() not in ["class", "state"]]
        colunas = comm.bcast(colunas, root=0)

        df = pd.concat([pd.read_parquet(f) for f in meus_arquivos])
        local = df.reindex(columns=colunas).to_numpy(dtype=np.float64)
        del df
        n_linhas = comm.allreduce(len(local), op=MPI.SUM)
        comm.Barrier()
        t_leitura = MPI.Wtime()

    comm.Barrier()
    t_distribuicao = MPI.Wtime()

    # ===== 2) Cálculo local + redução =====
    if args.calculo == "numpy":
        n_local, soma_local, min_local, max_local = somas_numpy(local)
    else:
        n_local, soma_local, min_local, max_local = somas_python(local)

    # Coluna toda NaN neste processo: usa ±infinito para não atrapalhar o MIN/MAX
    min_local[np.isnan(min_local)] = np.inf
    max_local[np.isnan(max_local)] = -np.inf

    # 1ª passada: N e soma de todos, para que TODOS os processos conheçam a média global
    n_global = np.zeros(len(colunas))
    soma_global = np.zeros(len(colunas))
    comm.Allreduce(n_local, n_global, op=MPI.SUM)
    comm.Allreduce(soma_local, soma_global, op=MPI.SUM)
    media = soma_global / n_global

    # 2ª passada: soma de (x - média)² com a média global (evita erro numérico da fórmula Σx²)
    if args.calculo == "numpy":
        desvios_local = desvios_numpy(local, media)
    else:
        desvios_local = desvios_python(local, media)

    min_global = np.zeros(len(colunas))
    max_global = np.zeros(len(colunas))
    desvios_global = np.zeros(len(colunas))
    comm.Reduce(min_local, min_global, op=MPI.MIN, root=0)
    comm.Reduce(max_local, max_global, op=MPI.MAX, root=0)
    comm.Reduce(desvios_local, desvios_global, op=MPI.SUM, root=0)

    comm.Barrier()
    t_fim = MPI.Wtime()

    # ===== 3) Resultado e tempos (só o Rank 0) =====
    if rank == 0:
        desvio = np.sqrt(desvios_global / (n_global - 1))
        resultado = pd.DataFrame({
            "n_amostras": n_global.astype(int),
            "media": media,
            "minimo": min_global,
            "maximo": max_global,
            "desvio_padrao": desvio,
        }, index=colunas)
        # Colunas sem amostras (ou com só 1, no caso do desvio) ficam como NaN
        resultado.loc[n_global == 0, ["media", "minimo", "maximo"]] = np.nan
        resultado.loc[n_global < 2, "desvio_padrao"] = np.nan

        os.makedirs("results", exist_ok=True)
        resultado.to_csv(f"results/resultado_mpi_{args.modo}_{args.calculo}_p{p}.csv")

        tempos = pd.DataFrame([{
            "modo": args.modo,
            "calculo": args.calculo,
            "p": p,
            "linhas": n_linhas,
            "t_leitura_s": round(t_leitura - t_inicio, 4),
            "t_distribuicao_s": round(t_distribuicao - t_leitura, 4),
            "t_calculo_reducao_s": round(t_fim - t_distribuicao, 4),
            "t_total_s": round(t_fim - t_inicio, 4),
        }])
        arquivo_tempos = "results/tempos_mpi.csv"
        tempos.to_csv(arquivo_tempos, mode="a", index=False,
                      header=not os.path.exists(arquivo_tempos))
        print(tempos.to_string(index=False))


if __name__ == "__main__":
    main()