import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def gerar_graficos():
    caminho_csv = os.path.join("results", "gabarito_referencia.csv")
    caminho_json = os.path.join("results", "meta_referencia.json")
    
    if not os.path.exists(caminho_csv) or not os.path.exists(caminho_json):
        print("Arquivos de resultado não encontrados em 'results/'. Rode 'python src/referencia.py' primeiro.")
        return

    df_gabarito = pd.read_csv(caminho_csv, index_col=0)
    with open(caminho_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Amostras temporais válidas por sensor
    df_gabarito['n_amostras'].plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title("Amostras Temporais Válidas (N) por Sensor", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Sensores", fontsize=10)
    ax1.set_ylabel("Quantidade de Amostras (N)", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Gráfico 2: Divisão do tempo (Leitura/IO vs Cálculo)
    tempos = [meta.get('tempo_carregamento_s', 0), meta.get('tempo_calculo_s', 0)]
    labels = [f"Leitura de Disco (E/S)\n{tempos[0]}s", f"Cálculo Matemático\n{tempos[1]}s"]
    colors = ['#ff9999', '#66b3ff']

    ax2.pie(tempos, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=(0.05, 0))
    ax2.set_title(f"Perfil de Execução Sequencial (Total T1: {meta.get('tempo_total_t1_s', 0)}s)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    caminho_img = os.path.join("results", "perfil_referencia.png")
    plt.savefig(caminho_img, dpi=300)
    print(f"Gráfico do perfil de execução salvo em: '{caminho_img}'")

if __name__ == "__main__":
    gerar_graficos()