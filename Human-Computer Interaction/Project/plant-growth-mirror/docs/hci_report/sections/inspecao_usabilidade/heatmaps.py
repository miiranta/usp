import matplotlib.pyplot as plt
import numpy as np

# Configurar estilo
plt.style.use('default')

# Dados das avaliações individuais (Nielsen)
nielsen_data = np.array([
    [1, 4, 5, 5, 5, 5, 5, np.nan, 5, 1],  # A01
    [3, 4, 4, 3, 5, 5, 2, 5, np.nan, 1],  # A02
    [5, 4, 4, 4, 3, 3, 3, 4, 2, 3],      # A03
    [3, np.nan, 5, 4, 5, 5, 5, 5, 5, np.nan],  # A04
    [3, 3, 4, 3, 3, 4, 4, 4, 3, np.nan]  # A05
])

# Dados das avaliações individuais (Norman)
norman_data = np.array([
    [5, 5, 5, 5, 5, 5],      # A01
    [4, 5, 4, 5, 5, 2],      # A02
    [4, 4, 5, 4, 3, 4],      # A03
    [5, 5, 3, 5, 5, 5],      # A04
    [np.nan, np.nan, 3, 4, 5, np.nan]  # A05
])

# Dados de problemas por heurística
nielsen_problems = np.array([12, 10, 11, 18, 13, 12, 9, 7, 5, 4])
norman_problems = np.array([8, 7, 6, 9, 5, 4])

# Labels
nielsen_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
norman_labels = ['1', '2', '3', '4', '5', '6']
avaliadores = ['A01', 'A02', 'A03', 'A04', 'A05']

# Criar figura com subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Heatmap das avaliações de Nielsen
im1 = ax1.imshow(nielsen_data, cmap='RdYlGn_r', aspect='auto')
ax1.set_title('Avaliações por Heurística de Nielsen', fontsize=14, fontweight='bold')
ax1.set_xlabel('Heurística')
ax1.set_ylabel('Avaliador')
ax1.set_xticks(range(len(nielsen_labels)))
ax1.set_xticklabels(nielsen_labels)
ax1.set_yticks(range(len(avaliadores)))
ax1.set_yticklabels(avaliadores)

# Adicionar valores no heatmap
for i in range(len(avaliadores)):
    for j in range(len(nielsen_labels)):
        if not np.isnan(nielsen_data[i, j]):
            ax1.text(j, i, f'{int(nielsen_data[i, j])}', 
                    ha='center', va='center', fontweight='bold', color='black')

plt.colorbar(im1, ax=ax1, label='Avaliação (1-5)')

# 2. Heatmap das avaliações de Norman
im2 = ax2.imshow(norman_data, cmap='RdYlGn_r', aspect='auto')
ax2.set_title('Avaliações por Princípio de Norman', fontsize=14, fontweight='bold')
ax2.set_xlabel('Princípio')
ax2.set_ylabel('Avaliador')
ax2.set_xticks(range(len(norman_labels)))
ax2.set_xticklabels(norman_labels)
ax2.set_yticks(range(len(avaliadores)))
ax2.set_yticklabels(avaliadores)

# Adicionar valores no heatmap
for i in range(len(avaliadores)):
    for j in range(len(norman_labels)):
        if not np.isnan(norman_data[i, j]):
            ax2.text(j, i, f'{int(norman_data[i, j])}', 
                    ha='center', va='center', fontweight='bold', color='black')

plt.colorbar(im2, ax=ax2, label='Avaliação (1-5)')

# 3. Gráfico de barras dos problemas por heurística de Nielsen
colors_nielsen = plt.cm.Set3(np.linspace(0, 1, len(nielsen_labels)))
bars1 = ax3.bar(nielsen_labels, nielsen_problems, color=colors_nielsen, alpha=0.8)
ax3.set_title('Problemas Detectados por Heurística de Nielsen', fontsize=14, fontweight='bold')
ax3.set_xlabel('Heurística')
ax3.set_ylabel('Número de Problemas')
ax3.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, value in zip(bars1, nielsen_problems):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value}', ha='center', va='bottom', fontweight='bold')

# 4. Gráfico de barras dos problemas por princípio de Norman
colors_norman = plt.cm.Set2(np.linspace(0, 1, len(norman_labels)))
bars2 = ax4.bar(norman_labels, norman_problems, color=colors_norman, alpha=0.8)
ax4.set_title('Problemas Detectados por Princípio de Norman', fontsize=14, fontweight='bold')
ax4.set_xlabel('Princípio')
ax4.set_ylabel('Número de Problemas')
ax4.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, value in zip(bars2, norman_problems):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('docs/hci_report/figures/heatmaps_usabilidade.png', dpi=300, bbox_inches='tight')
print("Gráfico heatmaps_usabilidade.png salvo com sucesso!")

# Criar gráfico adicional com distribuição de problemas por avaliador
fig2, ax5 = plt.subplots(1, 1, figsize=(10, 6))

avaliadores_problems = [23, 15, 18, 10, 16]
colors_avaliadores = plt.cm.viridis(np.linspace(0, 1, len(avaliadores)))

bars3 = ax5.bar(avaliadores, avaliadores_problems, color=colors_avaliadores, alpha=0.8)
ax5.set_title('Problemas Detectados por Avaliador', fontsize=14, fontweight='bold')
ax5.set_xlabel('Avaliador')
ax5.set_ylabel('Número de Problemas')
ax5.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, value in zip(bars3, avaliadores_problems):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('docs/hci_report/figures/problemas_por_avaliador.png', dpi=300, bbox_inches='tight')
print("Gráfico problemas_por_avaliador.png salvo com sucesso!") 