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

    # Cria a figura com mais espaço vertical
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Amostras temporais válidas por sensor
    df_gabarito['n_amostras'].plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title("Amostras Temporais Válidas (N) por Sensor", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Sensores", fontsize=10)
    ax1.set_ylabel("Quantidade de Amostras (N)", fontsize=10)
    ax1.tick_params(axis='x', rotation=90, labelsize=8) # Rotação de 90° e fonte ajustada
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Gráfico 2: Divisão do tempo (Leitura/IO vs Cálculo)
    t_io = meta.get('tempo_carregamento_s', 0)
    t_calc = meta.get('tempo_calculo_s', 0)
    t_total = meta.get('tempo_total_t1_s', 0)
    
    tempos = [t_io, t_calc]
    labels = [f"Leitura de Disco (E/S)\n{t_io}s", f"Cálculo Matemático\n{t_calc}s"]
    colors = ['#ff9999', '#66b3ff']

    ax2.pie(tempos, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=(0.05, 0))
    ax2.set_title(f"Perfil de Execução Sequencial (Total T1: {t_total}s)", fontsize=11, fontweight='bold')

    # Ajusta o espaçamento geral para não sobrepor nada
    plt.tight_layout()
    
    # Adiciona a legenda de hardware no topo/rodapé bem espaçado
    hardware_text = "Ambiente de Teste Sequencial (Baseline T1): Intel Core i5-11400F | 8GB RAM | Windows 11"
    fig.text(0.5, 0.02, hardware_text, ha='center', fontsize=9, fontstyle='italic', color='dimgray')

    plt.subplots_adjust(bottom=0.25) # Abre margem na parte inferior para os nomes dos sensores e legenda
    
    caminho_img = os.path.join("results", "perfil_referencia.png")
    plt.savefig(caminho_img, dpi=300)
    print(f"Gráfico corrigido e salvo em: '{caminho_img}'")

if __name__ == "__main__":
    gerar_graficos()