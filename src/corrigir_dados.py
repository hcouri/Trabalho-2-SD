import pandas as pd
import glob
import os

print("A iniciar a conversão dos ficheiros Parquet...")

caminho = "../3W/dataset/0/*.parquet"
ficheiros = glob.glob(caminho)

for f in ficheiros:
    print(f"A corrigir compressão e formato em: {os.path.basename(f)}")
    # O Pandas lê Brotli e nanossegundos sem qualquer problema
    df = pd.read_parquet(f)
    
    # Reescreve o ficheiro com compressão Snappy (nativa do Spark) e microssegundos
    df.to_parquet(
        f, 
        engine='pyarrow', 
        compression='snappy', 
        coerce_timestamps='us', 
        allow_truncated_timestamps=True
    )

print("\nConcluído! A base de dados está agora 100% compatível com o Apache Spark.")