import os
import time
import glob
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType


os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'



def processar_spark():
    # 1. Iniciar o motor Spark
    spark = SparkSession.builder \
        .appName("Processamento-3W-Pessoa3") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()

    print("A iniciar o PySpark e a contornar o problema dos nanossegundos...")
    inicio = time.time()

    caminho_base = "../3W/dataset/0"
    caminho_leitura = os.path.join(caminho_base, "*.parquet")

    # 2. O TRUQUE: Usar o Pandas para ler apenas 1 ficheiro e descobrir as colunas
    ficheiros_encontrados = glob.glob(caminho_leitura)
    if not ficheiros_encontrados:
        print("Erro: Nenhum ficheiro Parquet encontrado. Verifique o caminho.")
        return

    primeiro_ficheiro = ficheiros_encontrados[0]
    df_amostra = pd.read_parquet(primeiro_ficheiro)
    
    # Isolar apenas as colunas de sensores numéricos (ignorando textos, timestamps e a 'class')
    colunas_numericas = df_amostra.select_dtypes(include=[np.number]).columns.tolist()
    colunas_sensores = [col for col in colunas_numericas if col.lower() != 'class']

    # 3. Criar o esquema forçado para o Spark ler apenas essas colunas
    esquema_seguro = StructType([
        StructField(col, DoubleType(), True) for col in colunas_sensores
    ])
    
    print(f"Esquema seguro criado com {len(colunas_sensores)} sensores.")

    # 4. Ler com o Spark usando o esquema seguro (o Spark ignora completamente a coluna do tempo!)
    df = spark.read.schema(esquema_seguro).parquet(caminho_leitura)

    # 5. Montar as expressões matemáticas e executar
    expressoes = []
    for col in colunas_sensores:
        expressoes.append(F.mean(col).alias(f"{col}_media"))
        expressoes.append(F.min(col).alias(f"{col}_minimo"))
        expressoes.append(F.max(col).alias(f"{col}_maximo"))
        expressoes.append(F.stddev_samp(col).alias(f"{col}_desvio_padrao"))

    print("A calcular estatísticas globais no cluster. Isto pode demorar uns minutos...")
    
    # A magia distribuída acontece aqui:
    resultados_spark = df.agg(*expressoes).collect()[0].asDict()

    # 6. Formatar os resultados para coincidir com o gabarito do colega
    import math # adicione isto caso não tenha no topo
    
    estatisticas_formatadas = {}
    for col in colunas_sensores:
        # Pegamos nos valores calculados pelo Spark
        v_media = resultados_spark[f"{col}_media"]
        v_minimo = resultados_spark[f"{col}_minimo"]
        v_maximo = resultados_spark[f"{col}_maximo"]
        v_desvio = resultados_spark[f"{col}_desvio_padrao"]

        # Se for None (vazio), guardamos como NaN (Not a Number), senão convertemos para float
        estatisticas_formatadas[col] = {
            "media": float(v_media) if v_media is not None else float('nan'),
            "minimo": float(v_minimo) if v_minimo is not None else float('nan'),
            "maximo": float(v_maximo) if v_maximo is not None else float('nan'),
            "desvio_padrao": float(v_desvio) if v_desvio is not None else float('nan')
        }

    df_res = pd.DataFrame(estatisticas_formatadas).T
    
    os.makedirs("../results", exist_ok=True)
    caminho_saida = os.path.join("..", "results", "resultado_spark.csv")
    df_res.to_csv(caminho_saida)

    fim = time.time()
    
    print(f"\n--- Resultados do PySpark (Distribuído) ---")
    print(f"Tempo total de execução: {fim - inicio:.4f} segundos")
    print(f"Ficheiro guardado em: {caminho_saida}")
    
    spark.stop()

if __name__ == "__main__":
    processar_spark()