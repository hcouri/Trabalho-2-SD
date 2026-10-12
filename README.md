# Comparação entre MPI e Spark (Dataset 3W)

## Objetivo
Comparar o desempenho de duas implementações distribuídas — uma utilizando MPI com Python (`mpi4py`) e outra utilizando Apache Spark (`PySpark`) — no processamento de estatísticas descritivas sobre grandes volumes de dados de sensores.

## Aplicação Proposta
Processar matrizes de dados do dataset **3W 2.0.0 (Petrobras)**, contendo variáveis reais de poços de petróleo como pressão (`P-PDG`, `P-TP`), temperatura (`T-TP`), vazão e estados operacionais organizados em séries temporais (categorias `0` a `8`).

## Operações
Para cada variável numérica de sensor, calcular de forma exata:
- Número de amostras temporais válidas (N);
- Mínimo;
- Máximo;
- Média;
- Desvio Padrão amostral (ddof=1).

## Avaliação de Desempenho
- Medir o tempo de execução em diferentes quantidades de processos/executores;
- Calcular Speedup (S = T1 / Tp) e Eficiência (E = S / p);
- Avaliar três alterações no código que afetem o desempenho (E/S distribuída vs. centralizada, particionamento e vetorização);
- Comparar a eficiência entre as abordagens de MPI vs. Spark.

---

## 📁 Estrutura do Projeto

    Trabalho-2-SD/
    ├── src/
    │   └── referencia.py       # Algoritmo sequencial de referência (Pessoa 1)
    ├── tests/
    │   └── test_validacao.py   # Validação automática de resultados paralelos
    ├── results/
    │   ├── gabarito_referencia.csv  # Gabarito oficial de métricas
    │   └── meta_referencia.json     # Metadados de tempo (T1) e volume processado
    └── README.md

---

## Como Executar

### 1. Execução do Código de Referência (Sequencial - Pessoa 1)

O script `src/referencia.py` aceita argumentos via linha de comando para selecionar flexivelmente os casos/categorias do dataset:

* **Executar apenas o Caso 0 (Padrão para desenvolvimento):**
  ```bash
  python src/referencia.py

# Executar casos/categorias específicos (ex: Casos 0, 3 e 5):
# python src/referencia.py --categorias 0 3 5

# Executar para o dataset completo (Todas as categorias de 0 a 8):
# python src/referencia.py --categorias todas