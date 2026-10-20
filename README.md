# Comparação entre MPI e Spark (Dataset 3W)

## Objetivo
Comparar o desempenho de duas implementações distribuídas — uma utilizando MPI com Python (`mpi4py`) e outra utilizando Apache Spark (`PySpark`) — no processamento de estatísticas descritivas sobre grandes volumes de dados de sensores.

## Integrantes e Divisão de Tarefas
- **Pessoa 1 (Henrique Couri):** Aplicação e cálculos de referência, gabarito estatístico e streaming em lotes.
- **Pessoa 2 (Edgar Lutterbach):** Implementação paralela com MPI (`mpi4py`) usando primitivas coletivas (`Scatterv`, `Bcast`, `Allreduce`, `Reduce`).
- **Pessoa 3 (Vitto Baroni):** Implementação distribuída com Apache Spark (`PySpark`) e correção de esquema Parquet.
- **Pessoa 4 (Gabriel Ramalho):** Bateria de experimentos, escalabilidade ($p=1 \dots 128$), avaliação das 3 alterações arquiteturais, gráficos acadêmicos e relatório técnico.

## Operações Estatísticas
Para cada variável numérica de sensor no dataset **3W 2.0.0 (Petrobras)**:
- Quantidade de amostras temporais válidas ($N$);
- Média aritmética ($\mu$);
- Extremos operacionais (Mínimo e Máximo);
- Desvio Padrão amostral com correção de Bessel ($s$, $\text{ddof}=1$).

---

## 📁 Estrutura do Projeto

    Trabalho-2-SD/
    ├── src/
    │   ├── referencia.py             # Algoritmo sequencial de referência e gabarito (Pessoa 1)
    │   ├── estatisticas_mpi.py       # Implementação paralela MPI com mpi4py (Pessoa 2)
    │   ├── spark.py                  # Implementação distribuída PySpark (Pessoa 3)
    │   ├── corrigir_dados.py         # Utilitário de conversão de esquema Parquet (Pessoa 3)
    │   ├── metricas_desempenho.py    # Cálculo de Speedup e Eficiência (Pessoa 4)
    │   ├── benchmark.py              # Orquestrador de benchmarks de escalabilidade (Pessoa 4)
    │   ├── plot_escalabilidade.py    # Gerador de gráficos acadêmicos DPI 300 (Pessoa 4)
    │   └── plot_referencia.py        # Gráfico do perfil sequencial (Pessoa 1)
    ├── experiments/
    │   ├── exp_es_mpi.py             # Alteração 1: E/S Centralizada vs. Distribuída (Pessoa 4)
    │   ├── exp_spark_particionamento.py # Alteração 2: Particionamento no Spark (Pessoa 4)
    │   └── exp_vetorizacao.py        # Alteração 3: Vetorização em Memória (Pessoa 4)
    ├── tests/
    │   └── test_validacao.py         # Validação de fidelidade numérica contra o gabarito
    ├── results/
    │   ├── gabarito_referencia.csv   # Gabarito oficial de métricas
    │   ├── meta_referencia.json      # Metadados de tempo (T1) e hardware de referência
    │   ├── experimentos_escalabilidade.csv # Tabela com p=1..128, Speedup e Eficiência
    │   ├── tabela_escalabilidade.md  # Tabela formatada em Markdown
    │   ├── grafico_speedup.png       # Curva de Speedup MPI vs Spark vs Ideal
    │   ├── grafico_eficiencia.png    # Curva de Eficiência paralela
    │   ├── grafico_tempo_execucao.png# Tempo Tp vs p
    │   └── grafico_3_alteracoes.png  # Impacto das 3 alterações arquiteturais
    ├── docs/
    │   └── RELATORIO_TECNICO.md      # Artigo técnico completo de avaliação de desempenho
    └── README.md

---

## Como Executar

### 1. Execução do Código de Referência (Sequencial - Pessoa 1)
O script `src/referencia.py` processa o dataset sequencialmente e gera o gabarito oficial em `results/gabarito_referencia.csv`:
```bash
python src/referencia.py
```

### 2. Execução da Implementação MPI (Pessoa 2)
O script `src/estatisticas_mpi.py` aceita opções de modo de E/S e cálculo local:
```bash
# Execução padrão com 4 processos (E/S centralizada, cálculo vetorizado):
mpiexec -n 4 python src/estatisticas_mpi.py --categorias 0

# Execução com E/S distribuída (Alteração 1):
mpiexec -n 4 python src/estatisticas_mpi.py --categorias 0 --modo distribuido

# Execução com laços em Python nativo (Alteração 3):
mpiexec -n 4 python src/estatisticas_mpi.py --categorias 0 --calculo python
```

### 3. Execução da Implementação Apache Spark (Pessoa 3)
```bash
# Executar processamento distribuído PySpark:
python src/spark.py

# Caso necessário, converter arquivos Parquet para compatibilidade com Snappy:
python src/corrigir_dados.py
```

### 4. Validação Numérica dos Resultados (Pessoa 1, 2 e 3)
Verifica se as saídas do MPI ou Spark possuem erro absoluto inferior a $10^{-4}$ contra o gabarito:
```bash
python tests/test_validacao.py results/resultado_mpi_distribuido_numpy_p4.csv
python tests/test_validacao.py results/resultado_spark.csv
```

### 5. Experimentos, Benchmarks e Relatório Técnico (Pessoa 4)
Para executar a bateria de escalabilidade ($p = 1, 2, 4, 8, 16, 32, 64, 128$) e gerar todos os gráficos:
```bash
# Orquestrador de benchmarks de escalabilidade e métricas:
python src/benchmark.py --modo calibrado

# Avaliar as 3 alterações arquiteturais:
python src/benchmark.py --avaliar-alteracoes
python experiments/exp_es_mpi.py
python experiments/exp_spark_particionamento.py
python experiments/exp_vetorizacao.py

# Gerar todos os gráficos em alta resolução (DPI 300):
python src/plot_escalabilidade.py
```

O relatório acadêmico completo com análise teórica, gráficos e tabelas encontra-se em [`docs/RELATORIO_TECNICO.md`](docs/RELATORIO_TECNICO.md).