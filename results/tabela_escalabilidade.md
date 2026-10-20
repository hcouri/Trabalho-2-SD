# Tabela de Escalabilidade (Speedup e Eficiência)

| Tecnologia      |   Processos/Cores (p) | Tempo Médio (Tp)   | Speedup S(p)   | Eficiência E(p)   |
|:----------------|----------------------:|:-------------------|:---------------|:------------------|
| MPI (mpi4py)    |                     1 | 40.0055s           | 1.00x          | 100.0%            |
| Spark (PySpark) |                     1 | 40.5055s           | 0.99x          | 98.8%             |
| MPI (mpi4py)    |                     2 | 21.1353s           | 1.89x          | 94.6%             |
| Spark (PySpark) |                     2 | 23.0931s           | 1.73x          | 86.6%             |
| MPI (mpi4py)    |                     4 | 11.7101s           | 3.42x          | 85.4%             |
| Spark (PySpark) |                     4 | 14.2818s           | 2.80x          | 70.0%             |
| MPI (mpi4py)    |                     8 | 7.0314s            | 5.69x          | 71.1%             |
| Spark (PySpark) |                     8 | 10.0112s           | 4.00x          | 50.0%             |
| MPI (mpi4py)    |                    16 | 4.7531s            | 8.42x          | 52.6%             |
| Spark (PySpark) |                    16 | 8.1459s            | 4.91x          | 30.7%             |
| MPI (mpi4py)    |                    32 | 3.7242s            | 10.74x         | 33.6%             |
| Spark (PySpark) |                    32 | 7.7533s            | 5.16x          | 16.1%             |
| MPI (mpi4py)    |                    64 | 3.4084s            | 11.74x         | 18.3%             |
| Spark (PySpark) |                    64 | 8.6369s            | 4.63x          | 7.2%              |
| MPI (mpi4py)    |                   128 | 3.6084s            | 11.09x         | 8.7%              |
| Spark (PySpark) |                   128 | 11.2388s           | 3.56x          | 2.8%              |
