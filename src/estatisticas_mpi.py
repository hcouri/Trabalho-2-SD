import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mpi4py import MPI

# Reaproveita o carregamento da Pessoa 1, garantindo exatamente as mesmas linhas e colunas.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from referencia import carregar_dados_3w  # noqa: E402

RAIZ = 0
COLUNAS_EXCLUIDAS = {"class", "state"}  # mesmas exclusões da referência
LINHAS_POR_BLOCO = 500_000

def localizar_dataset(caminho_informado=None):
    """Procura a pasta 3W/dataset nos mesmos caminhos usados pela referência."""
    candidatos = [caminho_informado] if caminho_informado else [
        os.path.join("..", "3W", "dataset"),
        os.path.join("3W", "dataset"),
        os.path.join("..", "..", "3W", "dataset"),
    ]
    for caminho in candidatos:
        if caminho and os.path.exists(caminho):
            return caminho
    raise FileNotFoundError("Diretório '3W/dataset' não foi encontrado.")


def listar_arquivos(diretorio_dataset, categorias):
    """Mesma regra de busca de arquivos da referência (carregar_dados_3w)."""
    if "todas" in [str(c).lower() for c in categorias]:
        arquivos = glob.glob(os.path.join(diretorio_dataset, "*", "*.parquet"))
    else:
        arquivos = []
        for cat in categorias:
            arquivos.extend(glob.glob(os.path.join(diretorio_dataset, str(cat), "*.parquet")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo Parquet encontrado para: {categorias}")
    return sorted(arquivos)


def calcular_divisao(n_linhas, n_colunas, p):
    """
    Divide n_linhas entre p processos. Os primeiros 'resto' processos recebem 1 linha a mais.
    Retorna (linhas por processo, counts e displs em ELEMENTOS, como o Scatterv exige).
    """
    base, resto = divmod(n_linhas, p)
    linhas = np.array([base + 1 if r < resto else base for r in range(p)], dtype=np.int64)
    counts = linhas * n_colunas
    displs = np.zeros(p, dtype=np.int64)
    displs[1:] = np.cumsum(counts)[:-1]  # soma acumulada dos counts ANTERIORES
    return linhas, counts, displs


def repartir_arquivos(arquivos, tamanhos, p):
    """
    Rodízio "vai e volta" por tamanho (Saxena et al., 2020): ordena do maior para o menor
    e distribui 0,1,...,p-1, depois p-1,...,1,0, e assim por diante. Assim, quem recebe o
    maior arquivo de uma rodada recebe o menor da rodada seguinte, equilibrando a carga.
    """
    ordem = sorted(range(len(arquivos)), key=lambda i: tamanhos[i], reverse=True)
    partes = [[] for _ in range(p)]
    for posicao, i in enumerate(ordem):
        rodada, j = divmod(posicao, p)
        destino = j if rodada % 2 == 0 else p - 1 - j
        partes[destino].append(i)
    return partes


def acumular_primeira_passada(local):
    """Retorna, por coluna: N válidos, Σx, mínimo e máximo da fatia local (ignorando NaN)."""
    n_col = local.shape[1]
    n = np.zeros(n_col)
    soma = np.zeros(n_col)
    minimo = np.full(n_col, np.inf)    # ±inf: neutros para MIN/MAX caso a coluna
    maximo = np.full(n_col, -np.inf)   # seja toda NaN nesta fatia
    for inicio in range(0, local.shape[0], LINHAS_POR_BLOCO):
        bloco = local[inicio:inicio + LINHAS_POR_BLOCO]  # fatia de linhas = view, sem cópia
        n += (~np.isnan(bloco)).sum(axis=0)
        soma += np.nansum(bloco, axis=0)
        # fmin/fmax ignoram NaN (diferente de min/max, que propagam o NaN)
        minimo = np.fmin(minimo, np.fmin.reduce(bloco, axis=0, initial=np.inf))
        maximo = np.fmax(maximo, np.fmax.reduce(bloco, axis=0, initial=-np.inf))
    return n, soma, minimo, maximo


def acumular_desvios(local, media):
    """Retorna, por coluna, Σ(x - média)² da fatia local, usando a média GLOBAL."""
    desvios = np.zeros(local.shape[1])
    for inicio in range(0, local.shape[0], LINHAS_POR_BLOCO):
        bloco = local[inicio:inicio + LINHAS_POR_BLOCO]
        desvios += np.nansum((bloco - media) ** 2, axis=0)
    return desvios


# ======================================================================
# Fases de leitura + distribuição (uma por modo)
# ======================================================================
def ler_centralizado(comm, args):
    """Opção B: Rank 0 lê tudo e distribui fatias da matriz com Scatterv."""
    rank, p = comm.Get_rank(), comm.Get_size()

    matriz = None
    meta = None
    if rank == RAIZ:
        df = carregar_dados_3w(localizar_dataset(args.dataset), categorias=args.categorias)
        colunas = df.columns.tolist()
        # Matriz contígua em memória (ordem C = linha a linha), requisito para o Scatterv
        matriz = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
        del df
        meta = {"n_linhas": matriz.shape[0], "colunas": colunas}

    comm.Barrier()
    t_leitura = MPI.Wtime()

    # bcast minúsculo: objeto Python pequeno (serializado com pickle), custo desprezível.
    meta = comm.bcast(meta, root=RAIZ)
    n_col = len(meta["colunas"])
    linhas, counts, displs = calcular_divisao(meta["n_linhas"], n_col, p)

    # Scatterv maiúsculo: envia o buffer NumPy diretamente, sem pickle.
    if rank == RAIZ:
        # MPI.IN_PLACE: a fatia do próprio Rank 0 já está no início da matriz, então ele
        # não precisa copiá-la para outro buffer (com p = 1, isso evita uma cópia de 2,6 GB).
        comm.Scatterv([matriz, counts, displs, MPI.DOUBLE], MPI.IN_PLACE, root=RAIZ)
        fatia = matriz[:linhas[RAIZ]]
        # Com p > 1, copia só a fatia (pequena) e libera a matriz inteira da memória.
        # Com p = 1, a fatia É a matriz inteira, então a usa diretamente, sem cópia.
        local = fatia if p == 1 else fatia.copy()
        del fatia
    else:
        local = np.empty((linhas[rank], n_col), dtype=np.float64)
        comm.Scatterv(None, local, root=RAIZ)
    del matriz

    comm.Barrier()
    t_distribuicao = MPI.Wtime()
    return local, meta, t_leitura, t_distribuicao


def ler_distribuido(comm, args):
    """Opção A: cada processo lê diretamente o seu conjunto de arquivos Parquet."""
    rank, p = comm.Get_rank(), comm.Get_size()

    # --- Distribuição: o Rank 0 só lê METADADOS (rápido), não os dados ---
    partes = None
    meta = None
    if rank == RAIZ:
        arquivos = listar_arquivos(localizar_dataset(args.dataset), args.categorias)
        print(f"Total de arquivos Parquet localizados para as categorias "
              f"{args.categorias}: {len(arquivos)}")
        n_linhas_arq, colunas, vistas = [], [], set()
        for f in arquivos:
            md = pq.read_metadata(f)
            n_linhas_arq.append(md.num_rows)
            # Mesma regra da referência: colunas numéricas, exceto class/state, na ordem
            # em que aparecem pela 1ª vez (o pd.concat da referência preserva essa ordem).
            esquema = md.schema.to_arrow_schema()
            # Colunas que o pandas gravou como ÍNDICE (ex.: timestamp) não são sensores:
            # o pd.read_parquet da referência as transforma em índice, não em coluna.
            indices_pandas = {c for c in (esquema.pandas_metadata or {}).get("index_columns", [])
                              if isinstance(c, str)}
            for campo in esquema:
                numerico = pa.types.is_floating(campo.type) or pa.types.is_integer(campo.type)
                if (numerico and campo.name not in vistas and campo.name not in indices_pandas
                        and campo.name.lower() not in COLUNAS_EXCLUIDAS):
                    vistas.add(campo.name)
                    colunas.append(campo.name)
        tamanhos = [os.path.getsize(f) for f in arquivos]
        indices = repartir_arquivos(arquivos, tamanhos, p)
        partes = [[(arquivos[i], n_linhas_arq[i]) for i in idx] for idx in indices]
        meta = {"n_linhas": int(sum(n_linhas_arq)), "colunas": colunas}

    # scatter minúsculo: cada processo recebe a SUA lista de (arquivo, nº de linhas).
    meus_arquivos = comm.scatter(partes, root=RAIZ)
    meta = comm.bcast(meta, root=RAIZ)
    colunas = meta["colunas"]

    comm.Barrier()
    t_distribuicao = MPI.Wtime()

    # --- Leitura paralela: cada processo lê só os seus arquivos ---
    # A matriz local é pré-alocada com o total de linhas, evitando juntar pedaços depois
    # (o que duplicaria a memória).
    total_local = sum(n for _, n in meus_arquivos)
    local = np.empty((total_local, len(colunas)), dtype=np.float64)
    pos = 0
    for f, n in meus_arquivos:
        presentes = [c for c in colunas if c in pq.read_schema(f).names]
        df = pd.read_parquet(f, columns=presentes).reindex(columns=colunas)
        local[pos:pos + n] = df.to_numpy(dtype=np.float64, na_value=np.nan)
        pos += n
        del df

    comm.Barrier()
    t_leitura = MPI.Wtime()
    return local, meta, t_leitura, t_distribuicao


# ======================================================================
# Programa principal
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Estatísticas do 3W com MPI (mpi4py)")
    parser.add_argument("--categorias", nargs="+", default=["0"])
    parser.add_argument("--modo", choices=["centralizado", "distribuido"],
                        default="centralizado", help="Estratégia de E/S (Alteração 1)")
    parser.add_argument("--dataset", default=None, help="Caminho da pasta 3W/dataset")
    parser.add_argument("--saida", default="results", help="Pasta de saída")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    comm.Barrier()
    t_inicio = MPI.Wtime()

    # ------------------------------------------------------------------
    # Fases 1 e 2 - Leitura e distribuição (a ordem das fases depende do modo)
    # ------------------------------------------------------------------
    if args.modo == "centralizado":
        local, meta, t_leitura, t_distribuicao = ler_centralizado(comm, args)
        dur_leitura = t_leitura - t_inicio
        dur_distribuicao = t_distribuicao - t_leitura
        t_dados_prontos = t_distribuicao
    else:
        local, meta, t_leitura, t_distribuicao = ler_distribuido(comm, args)
        dur_distribuicao = t_distribuicao - t_inicio
        dur_leitura = t_leitura - t_distribuicao
        t_dados_prontos = t_leitura

    colunas = meta["colunas"]
    n_col = len(colunas)

    # ------------------------------------------------------------------
    # Fase 3 - Cálculo local + redução
    # ------------------------------------------------------------------
    # 1ª passada: N e Σx locais (ignorando NaN), somados entre todos com Allreduce.
    # Os dois vetores vão juntos em um único buffer para fazer UMA chamada em vez de duas.
    n_local, somax_local, min_local, max_local = acumular_primeira_passada(local)
    soma_local = np.stack([n_local, somax_local])  # linha 0 = N, linha 1 = Σx
    soma_global = np.empty_like(soma_local)
    comm.Allreduce(soma_local, soma_global, op=MPI.SUM)

    n_global = soma_global[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        media_global = soma_global[1] / n_global  # todos os processos conhecem a média

    # 2ª passada: Σ(x - média)² local, usando a média GLOBAL.
    desvios_local = acumular_desvios(local, media_global)

    min_global = np.empty(n_col) if rank == RAIZ else None
    max_global = np.empty(n_col) if rank == RAIZ else None
    desvios_global = np.empty(n_col) if rank == RAIZ else None
    comm.Reduce(min_local, min_global, op=MPI.MIN, root=RAIZ)
    comm.Reduce(max_local, max_global, op=MPI.MAX, root=RAIZ)
    comm.Reduce(desvios_local, desvios_global, op=MPI.SUM, root=RAIZ)

    comm.Barrier()
    t_fim = MPI.Wtime()

    # ------------------------------------------------------------------
    # Fase 4 - Montagem do resultado e registro dos tempos (só o Rank 0)
    # ------------------------------------------------------------------
    if rank != RAIZ:
        return

    vazio = n_global == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        desvio = np.sqrt(desvios_global / (n_global - 1))
    desvio[n_global < 2] = np.nan
    media_global[vazio] = np.nan
    min_global[vazio] = np.nan
    max_global[vazio] = np.nan

    os.makedirs(args.saida, exist_ok=True)
    caminho_csv = os.path.join(args.saida, f"resultado_mpi_{args.modo}_p{p}.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["", "n_amostras", "media", "minimo", "maximo", "desvio_padrao"])
        for i, col in enumerate(colunas):
            # repr(float(...)) grava o valor com precisão total (17 dígitos significativos)
            escritor.writerow([col, int(n_global[i]), repr(float(media_global[i])),
                               repr(float(min_global[i])), repr(float(max_global[i])),
                               repr(float(desvio[i]))])

    tempos = {
        "modo": args.modo,
        "p": p,
        "categorias": " ".join(args.categorias),
        "linhas": meta["n_linhas"],
        "colunas": n_col,
        "t_leitura_s": round(dur_leitura, 4),
        "t_distribuicao_s": round(dur_distribuicao, 4),
        "t_calculo_reducao_s": round(t_fim - t_dados_prontos, 4),
        "t_total_s": round(t_fim - t_inicio, 4),
    }

    # Acumula uma linha por execução: é a tabela que a Pessoa 4 usa para Speedup/Eficiência.
    caminho_tempos = os.path.join(args.saida, "tempos_mpi.csv")
    novo = not os.path.exists(caminho_tempos)
    with open(caminho_tempos, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(tempos.keys()))
        if novo:
            escritor.writeheader()
        escritor.writerow(tempos)

    print(json.dumps(tempos, indent=4, ensure_ascii=False))
    print(f"Resultado salvo em '{caminho_csv}'")


if __name__ == "__main__":
    main()