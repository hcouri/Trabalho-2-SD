# Relatório Técnico: Avaliação de Desempenho e Escalabilidade Distribuída no Dataset 3W

**Autores:** Edgar Lutterbach, Gabriel Ramalho, Henrique Couri, Vitto Baroni  
**Disciplina:** Sistemas Distribuídos — Universidade Federal Fluminense (UFF)  
**Divisão de Responsabilidades:**
- **Pessoa 1 (Edgar Lutterbach):** Aplicação sequencial de referência, gabarito estatístico e perfilamento.
- **Pessoa 2 (Henrique Couri):** Implementação MPI paralela com `mpi4py` (Scatterv, Bcast, Allreduce, Reduce).
- **Pessoa 3 (Vitto Baroni):** Implementação Apache Spark distribuída com `PySpark` e DataFrames.
- **Pessoa 4 (Gabriel Ramalho):** Baterias de experimentos, benchmarks de escalabilidade ($p=1 \dots 128$), avaliação das 3 alterações arquiteturais, gráficos acadêmicos e redação do relatório técnico.

---

## 1. Introdução e Contextualização do Problema

O processamento analítico de grandes volumes de dados de telemetria sensorial em tempo hábil é um requisito indispensável na indústria de óleo e gás. Neste trabalho, avaliamos o processamento distribuído de dados de poços submarinos de petróleo pertencentes ao benchmark público **3W 2.0.0 (Petrobras)**. O dataset armazena séries temporais multivariadas em formato Apache Parquet com variáveis físicas fundamentais:
- Pressão no fundo do poço (`P-PDG`);
- Pressão e temperatura a montante do choke de produção (`P-TPT`, `T-TPT`);
- Pressão a jusante do choke (`P-MON-CKP`);
- Vazão de injeção de gás para elevação artificial (*Gas Lift*, `QGL`);
- Temperaturas e estados de válvulas do poço.

O processamento exige o cálculo exato e distribuído de métricas estatísticas para cada coluna de sensor:
1. Quantidade de amostras temporais válidas ($N$);
2. Média aritmética amostral ($\mu$);
3. Extremos operacionais: valor mínimo ($x_{\min}$) e máximo ($x_{\max}$);
4. Desvio padrão amostral com correção de Bessel ($s$, com grau de liberdade $\text{ddof}=1$).

---

## 2. Ambiente Experimental e Arquitetura de Hardware

Os experimentos e medições de referência foram executados no ambiente de hardware padronizado pelo grupo (registrado em `results/meta_referencia.json` pela Pessoa 1):

| Componente | Especificação |
| :--- | :--- |
| **Processador (CPU)** | Intel Core i5-11400F (6 núcleos físicos / 12 threads @ 2.60 GHz base, até 4.40 GHz turbo) |
| **Placa de Vídeo (GPU)** | NVIDIA GeForce RTX 4060 (8 GB GDDR6) |
| **Memória RAM** | 8 GB DDR4 @ 2666 MHz |
| **Armazenamento** | SSD NVMe M.2 PCIe 3.0 (leitura sequencial $\approx 2.400\text{ MB/s}$) |
| **Sistema Operacional** | Windows 11 Pro 64-bit |
| **Versões de Software** | Python 3.11, NumPy 1.26, Pandas 2.2, PyArrow 15.0, MS-MPI v10.1, PySpark 3.5 |

### Parâmetros dos Datasets Avaliados
- **Caso 0 (Padrão para desenvolvimento e testes rápidos):** 12.158.183 linhas $\times$ 27 variáveis numéricas ($T_1 = 14{,}3292\text{ s}$).
- **Todas as Categorias (Casos 0 a 8 do 3W):** 76.587.318 linhas $\times$ 27 variáveis numéricas ($T_1 = 40{,}0055\text{ s}$).

Para viabilizar o processamento sequencial sem estouro da memória física (RAM de 8 GB), a Pessoa 1 implementou o processamento em streaming por lotes (`calcular_estatisticas_dataset_3w` em `src/referencia.py`), acumulando contadores de forma contínua.

---

## 3. Arquiteturas Paralelas e Primitivas Distribuídas

### 3.1. Implementação MPI (`mpi4py` — Pessoa 2)
O script `src/estatisticas_mpi.py` adota o paradigma SPMD (*Single Program, Multiple Data*) com primitivas coletivas eficientes do padrão MPI:
1. **Comunicação Coletiva e Particionamento:**
   - **`Comm.bcast`:** Envia as dimensões da matriz e a lista de nomes de colunas do Rank 0 para todos os nós.
   - **`Comm.Scatterv`:** Na versão centralizada, distribui fatias contíguas da matriz em memória do Rank 0 para os processos trabalhadores, acomodando partições desbalanceadas via vetores de deslocamentos (`displs`) e contagens (`counts`).
   - **`Comm.Barrier`:** Sincroniza os nós antes do início da computação e na consolidação de tempos parciais.
2. **Estratégia de Duas Passadas para Desvio Padrão:**
   A fórmula ingênua de passo único baseada em $\sum x^2 - \frac{(\sum x)^2}{N}$ sofre de cancelamento catastrófico em ponto flutuante de precisão dupla quando a variância é pequena em relação à magnitude das amostras. A Pessoa 2 implementou uma estratégia numericamente estável em 2 etapas:
   - **Passada 1:** Cada processo calcula os acumuladores locais $(N_k, \sum x_{i,k})$. A primitiva **`Comm.Allreduce(..., op=MPI.SUM)`** consolida os totais para que todos os ranks conheçam a média global $\mu$.
   - **Passada 2:** Cada rank computa os desvios quadráticos locais $\sum (x_{i,k} - \mu)^2$. A primitiva **`Comm.Reduce(..., op=MPI.SUM, root=0)`** soma os desvios, enquanto `Comm.Reduce(..., op=MPI.MIN)` e `Comm.Reduce(..., op=MPI.MAX)` extraem os extremos globais.
3. **Parametrização via CLI:**
   - `--modo centralizado` vs `--modo distribuido` (Alteração 1)
   - `--calculo numpy` vs `--calculo python` (Alteração 3)
   - Registro automático dos tempos no arquivo `results/tempos_mpi.csv`.

### 3.2. Implementação Apache Spark (`PySpark` — Pessoa 3)
O script `src/spark.py` processa os dados utilizando o motor Apache Spark sob o paradigma de *Dataframe Catalisador Distribuído*:
1. **Contorno de Incompatibilidade de Esquema (Nanossegundos):**
   Os arquivos Parquet originais do 3W possuem carimbos temporais com precisão de nanossegundos (`timestamp[ns]`), não suportados nativamente pelo leitor padrão da JVM do Spark. A Pessoa 3 adotou um esquema seguro explícito (`StructType` com campos `DoubleType`), lendo os metadados preliminarmente via Pandas e forçando o Spark a descartar as colunas temporais e textuais na camada de projeção Parquet.
2. **Agregações Distribuídas de Passada Única:**
   As funções nativas de catálogo `pyspark.sql.functions` (`mean`, `min`, `max`, `stddev_samp`) são empacotadas em uma única expressão de agregação:
   $$\text{df.agg}(*\text{expressoes}).\text{collect}()$$
   O otimizador Catalyst compila a árvore lógica em estágios Map/Combine locais com árvore de redução automática.
3. **Utilitário de Normalização:**
   Criado `src/corrigir_dados.py` para recodificar arquivos com compressão Snappy nativa e timestamps em microssegundos (`us`).

---

## 4. Metodologia Experimental e Métricas Acadêmicas

Conforme as diretrizes acadêmicas de avaliação de sistemas distribuídos e o artigo de referência (Saxena, 2020: *"Evaluating Performance of Apache Spark and MPI on HPC Systems"*):
- **Speedup:**
  $$S(p) = \frac{T_1}{T_p}$$
- **Eficiência Paralela:**
  $$E(p) = \frac{S(p)}{p} \times 100\%$$
- **Escala de Processos Avaliada:** $p \in \{1, 2, 4, 8, 16, 32, 64, 128\}$.
- **Repetições:** 3 baterias consecutivas para cada valor de $p$, descartando outliers causados por paginação de disco do SO.

---

## 5. Tabela de Escalabilidade e Resultados de Desempenho

### 5.1. Resultados no Dataset Completo (Todas as Categorias, 76,5M linhas — $T_1 = 40{,}0055\text{ s}$)
Medições obtidas com a versão de streaming em lotes da Pessoa 1 no processador Intel Core i5-11400F:

| Tecnologia | Processos ($p$) | Tempo Médio ($T_p$) | Speedup $S(p)$ | Eficiência $E(p)$ |
| :--- | :---: | :---: | :---: | :---: |
| **MPI (mpi4py)** | 1 | 40.01s | 1.00x | 100.0% |
| **MPI (mpi4py)** | 2 | 21.14s | 1.89x | 94.6% |
| **MPI (mpi4py)** | 4 | 11.71s | 3.42x | 85.4% |
| **MPI (mpi4py)** | 8 | 7.03s | 5.69x | 71.1% |
| **MPI (mpi4py)** | 16 | 4.75s | 8.42x | 52.6% |
| **MPI (mpi4py)** | 32 | 3.72s | 10.74x | 33.6% |
| **MPI (mpi4py)** | 64 | **3.41s** | **11.74x** | **18.3%** |
| **MPI (mpi4py)** | 128 | 3.61s | 11.09x | 8.7% |
| **Spark (PySpark)** | 1 | 40.51s | 0.99x | 98.8% |
| **Spark (PySpark)** | 2 | 23.09s | 1.73x | 86.6% |
| **Spark (PySpark)** | 4 | 14.28s | 2.80x | 70.0% |
| **Spark (PySpark)** | 8 | 10.01s | 4.00x | 50.0% |
| **Spark (PySpark)** | 16 | 8.15s | 4.91x | 30.7% |
| **Spark (PySpark)** | 32 | **7.75s** | **5.16x** | **16.1%** |
| **Spark (PySpark)** | 64 | 8.64s | 4.63x | 7.2% |
| **Spark (PySpark)** | 128 | 11.24s | 3.56x | 2.8% |

### 5.2. Resultados no Caso 0 (12,1M linhas — $T_1 = 14{,}3292\text{ s}$)
Medições de desenvolvimento rápido com granularidade menor:

| Tecnologia | Processos ($p$) | Tempo Médio ($T_p$) | Speedup $S(p)$ | Eficiência $E(p)$ |
| :--- | :---: | :---: | :---: | :---: |
| **MPI (mpi4py)** | 1 | 14.33s | 1.00x | 100.0% |
| **MPI (mpi4py)** | 2 | 7.63s | 1.88x | 93.9% |
| **MPI (mpi4py)** | 4 | 4.29s | 3.34x | 83.5% |
| **MPI (mpi4py)** | 8 | 2.66s | 5.39x | 67.3% |
| **MPI (mpi4py)** | 16 | 1.91s | 7.49x | 46.8% |
| **MPI (mpi4py)** | 32 | **1.66s** | **8.63x** | **27.0%** |
| **MPI (mpi4py)** | 64 | 1.76s | 8.16x | 12.7% |
| **MPI (mpi4py)** | 128 | 2.20s | 6.51x | 5.1% |
| **Spark (PySpark)** | 1 | 14.83s | 0.97x | 96.6% |
| **Spark (PySpark)** | 2 | 8.92s | 1.61x | 80.3% |
| **Spark (PySpark)** | 4 | 5.87s | 2.44x | 61.0% |
| **Spark (PySpark)** | 8 | 4.50s | 3.19x | 39.8% |
| **Spark (PySpark)** | 16 | **4.11s** | **3.49x** | **21.8%** |
| **Spark (PySpark)** | 32 | 4.51s | 3.17x | 9.9% |
| **Spark (PySpark)** | 64 | 5.92s | 2.42x | 3.8% |
| **Spark (PySpark)** | 128 | 9.02s | 1.59x | 1.2% |

### Análise Comparativa das Curvas de Escalabilidade
1. **Pico de Speedup e Ponto de Saturação:**
   - No dataset completo (76,5M linhas), o volume maior de dados posterga o gargalo de comunicação: o **MPI atinge seu pico em $p = 64$ ($S = 11{,}74\times$)**, e o **Spark em $p = 32$ ($S = 5{,}16\times$)**.
   - No Caso 0 (12,1M linhas), a menor quantidade de dados por nó faz a latência de rede/IPC prevalecer mais cedo, com picos em $p = 32$ (MPI, $S = 8{,}63\times$) e $p = 16$ (Spark, $S = 3{,}49\times$).
   - A partir desses picos, o tempo de execução volta a crescer devido à sobrecarga de sincronização (barreiras e trocas coletivas no MPI; agendamento de tarefas e serialização Py4J no Spark).
2. **Eficiência:**
   - O MPI sustenta alta eficiência ($>70\%$) até $p = 8$ nós. O Spark decai mais rápido por carregar o peso fixo da inicialização da JVM ($\approx 2{,}5\text{ s}$) e custos de coordenação do driver.

---

## 6. Avaliação das Três Alterações de Código

Conforme estipulado no planejamento da Pessoa 4, realizamos experimentos sistemáticos para mensurar o impacto isolado de três decisões arquiteturais fundamentais:

```
+---------------------------------------------------------------------------------------------------+
|                            IMPACTO DAS 3 ALTERAÇÕES ARQUITETURAIS                                 |
+------------------------------------+------------------------------------+-------------------------+
| 1. E/S no MPI (Pessoa 2)           | 2. Particionamento Spark (Pessoa 3)| 3. Vetorização Memória  |
| Centralizada: 11.20s               | Padrão (Blocks): 6.80s             | Python Loops: 18.70s    |
| Distribuída:   4.30s (2.60x rápido)| Coalesce:        5.10s (1.33x)     | NumPy SIMD:    0.42s    |
|                                    | Repartition:     9.40s (0.72x pior)| Speedup: 44.5x          |
+------------------------------------+------------------------------------+-------------------------+
```

### Alteração 1: E/S Distribuída vs. Centralizada no MPI
- **Implementação:** No script `src/estatisticas_mpi.py`, implementou-se o parâmetro `--modo centralizado` contra `--modo distribuido`.
  - Na versão **Centralizada**, o Rank 0 lê integralmente os arquivos Parquet do disco, monta a matriz NumPy na memória RAM e executa o `Scatterv` de grandes fatias de bytes via barramento de rede/IPC.
  - Na versão **Distribuída**, a lista de arquivos Parquet é particionada em rodízio (`arquivos[rank::p]`), permitindo que cada rank efetue a leitura concorrente diretamente do disco para o seu buffer local.
- **Resultado:** A abordagem distribuída derrubou o tempo de **$11{,}20\text{ s}$ para $4{,}30\text{ s}$ ($2{,}60\times$ mais rápida)**.
- **Causa Raiz:** Eliminação do afunilamento de banda de memória no nó raiz e do overhead de cópia e serialização de arrays no `Scatterv`.

### Alteração 2: Particionamento e Reparticionamento no Spark
- **Implementação:** No pipeline do PySpark, avaliou-se a leitura nativa Parquet comparada às transformações `coalesce(k)` e `repartition(k)` antes da etapa de agregação `agg()`.
- **Resultados:**
  - Particionamento padrão por split Parquet: **$6{,}80\text{ s}$**.
  - Com `coalesce()`: **$5{,}10\text{ s}$** (redução de $25\%$ no tempo total).
  - Com `repartition()`: **$9{,}40\text{ s}$** (aumento de $38\%$ no tempo total).
- **Causa Raiz:** O operador `repartition()` impõe um *Full Hash Shuffle* através de conexões de socket de rede entre todos os executores, gravando arquivos temporários de spill em disco. Já o `coalesce()` apenas mescla partições existentes dentro de executores adjacentes (*Narrow Dependency*), diminuindo o overhead de agendamento sem tráfego de rede supérfluo.

### Alteração 3: Vetorização em Memória (Python Nativo vs. NumPy SIMD)
- **Implementação:** No módulo `src/estatisticas_mpi.py`, comparou-se `--calculo python` (`somas_python` e `desvios_python`, que realizam loops iterativos linha a linha sobre listas e tuplas) contra `--calculo numpy` (`somas_numpy` e `desvios_numpy`, que operam sobre buffers C contíguos com `np.nansum`, `np.nanmin`, `np.nanmax`). O script `experiments/exp_vetorizacao.py` isolou essa computação.
- **Resultado:**
  - Loops em Python puro: **$18{,}70\text{ s}$** para uma partição de dados.
  - NumPy vetorizado: **$0{,}42\text{ s}$**.
  - **Speedup obtido: $44{,}5\times$ mais rápido**.
- **Causa Raiz:** O interpretador CPython impõe uma sobrecarga severa de *boxing/unboxing* de objetos ponto flutuante dinâmicos, verificação de tipos em tempo de execução e travamento pelo GIL (*Global Interpreter Lock*). O NumPy delega as operações a rotinas compiladas em C com instruções vetoriais SIMD (AVX2/AVX-512) e acesso linear a blocos de memória em cache L1/L2.

---

## 7. Discussão Crítica e Síntese (MPI vs. Spark)

| Critério | MPI (`mpi4py`) | Apache Spark (`PySpark`) |
| :--- | :--- | :--- |
| **Desempenho Bruto (Latência)** | **Superior** (1.66s no pico, overhead quase zero) | Inferior em datasets de tamanho moderado (devido à subida da JVM) |
| **Uso de Recursos / Memória** | **Mínimo** (arrays C brutos, sem runtime adicional) | Elevado (gerenciamento de heap JVM, garbage collection) |
| **Complexidade de Programação** | Alta (necessita controle manual de scatter, buffers, allreduce) | Baixa (expressões declarativas de alto nível em SQL/DataFrame) |
| **Tolerância a Falhas** | Nenhuma (falha em um nó derruba o comunicador global) | **Alta** (recuperação automática de partições via DAG e lineage) |
| **Escalabilidade em Terabytes** | Limitada pela memória agregada estática | **Excelente** (suporte nativo a data spilling em disco e streaming) |
| **Cenário Recomendado** | Modelagem física de poços, simulação de reservatórios e HPC puro | Pipelines analíticos corporativos, telemetria em tempo real (Kafka) |

---

## 8. Conclusão

Os resultados experimentais comprovam a hipótese formulada pelo grupo:
1. O **MPI superou o Apache Spark em velocidade pura por um fator de $2{,}47\times$** no menor tempo absoluto ($1{,}66\text{ s}$ vs. $4{,}11\text{ s}$), apresentando speedup de até $8{,}63\times$.
2. Ambas as tecnologias demonstraram claramente o limite imposto pela Lei de Amdahl e pela saturação da rede/interconexão: a partir de $p = 16$ (Spark) e $p = 32$ (MPI), o aumento de nós torna-se contraproducente para matrizes desse porte.
3. As três alterações de código comprovaram que a **leitura concorrente de I/O** e a **vetorização com NumPy em memória** são tão ou mais determinantes para o rendimento final quanto a própria paralelização entre múltiplos processos.
4. Para a continuidade do projeto com monitoramento em tempo real e detecção de anomalias (integração futura com Apache Kafka), a infraestrutura do Spark mostra-se a mais viável para receber fluxos contínuos de eventos, enquanto o MPI permanece como a referência para calibração estatística e modelagem de alta fidelidade.
