# Comparação entre MPI e Spark

## Objetivo

Comparar o desempenho de duas implementações distribuídas, uma utilizando MPI com Python (`mpi4py`) e outra utilizando Apache Spark (`PySpark`).

## Aplicação proposta

Processar uma matriz de dados simulados relacionados à produção de petróleo, contendo variáveis como pressão, temperatura, vazão e produção.

## Operações

Para cada variável, calcular:
- Mínimo;
- Máximo;
- Média;
- Desvio padrão.

## Avaliação de desempenho

- Medir o tempo de execução com diferentes quantidades de processos;
- Calcular speedup e eficiência;
- Avaliar três alterações no código que possam afetar o desempenho;
- Comparar os resultados das implementações MPI e Spark.

## Estrutura do projeto

- `src/`: implementações e código de referência;
- `tests/`: testes de validação;
- `results/`: resultados dos experimentos.
